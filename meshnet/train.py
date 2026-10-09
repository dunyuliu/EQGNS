import sys
import os
import glob
import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
import torch_geometric.transforms as T
import re
import pickle
from tqdm import tqdm

from absl import flags
from absl import app

import json 

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from meshnet import data_loader
from meshnet import learned_simulator
from meshnet import seeding
from meshnet.fast_rollout import rollout_fast
from meshnet.noise import get_velocity_noise
from meshnet.utils import datas_to_graph
from meshnet.utils import NodeType
from meshnet.utils import optimizer_to


flags.DEFINE_enum(
    'mode', 'train', ['train', 'valid', 'rollout'],
    help='Train model, validation or rollout evaluation.')
flags.DEFINE_integer('batch_size', 2, help='The batch size.')
flags.DEFINE_string('data_path', None, help='The dataset directory.')
flags.DEFINE_string('model_path', "model/", help=('The path for saving checkpoints of the model.'))
flags.DEFINE_string('output_path', "rollouts/", help='The path for saving outputs (e.g. rollouts).')
flags.DEFINE_string('model_file', None, help=('Model filename (.pt) to resume from. Can also use "latest" to default to newest file.'))
flags.DEFINE_string('train_state_file', None, help=('Train state filename (.pt) to resume from. Can also use "latest" to default to newest file.'))
flags.DEFINE_integer("cuda_device_number", None, help="CUDA device (zero indexed), default is None so default CUDA device will be used.")
flags.DEFINE_string('rollout_filename', "rollout", help='Name saving the rollout')
flags.DEFINE_integer('ntraining_steps', int(1E7), help='Number of training steps.')
flags.DEFINE_integer('nsave_steps', int(5000), help='Number of steps at which to save the model.')
flags.DEFINE_integer('seed', None, help=(
    'Opt-in RNG seed (meshnet/seeding.py). Default None: unseeded, byte-identical to '
    'pre-seeding behaviour. When set, model init, training/validation sample order, and '
    'training noise are each seeded from an independent sub-stream derived from this seed, '
    'and their RNG state is checkpointed in train_state-<step>.pt for exact resume.'))
flags.DEFINE_integer('rollout_batch_size', 1, help=(
    'Number of same-length test trajectories rolled out together as one disjoint graph. '
    'Default 1: the original one-trajectory-at-a-time path, bit-identical to before. '
    '>1: same math per trajectory, but rounding differs and long rollouts can diverge during '
    'active rupture; use for evaluation, not for the paper-parity gate '
    '(docs/user/rollout_and_analysis.md, "Batched rollout").'))
flags.DEFINE_enum('rollout_fast', 'off', ['off', 'fp32', 'tf32', 'fp16', 'bf16'], help=(
    'Opt-in fast rollout (meshnet/fast_rollout.py): static-graph restructured GNN, torch.compile and '
    'a CUDA graph per step, with fp32 / tf32 / fp16 / bf16 matmuls. Default off: the original path. '
    'Rounding differs, so use for evaluation, not for the paper-parity gate (docs/user/rollout_and_analysis.md).'))
flags.DEFINE_boolean('deterministic', False, help=(
    'Only meaningful with --seed set. Additionally asks torch for deterministic kernels '
    '(warn-only); see meshnet/seeding.py:set_deterministic.'))
FLAGS = flags.FLAGS

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = torch.device('cpu')
# an instance that transforms face-based graph to edge-based graph. Edge features are auto-computed using "Cartesian" and "Distance"
transformer = T.Compose([T.FaceToEdge(), T.Cartesian(norm=False), T.Distance(norm=False)])


def compute_edge_features(node_coords, edge_index):
    """Compute edge features (Cartesian + Distance) from node positions."""
    row, col = edge_index
    cart = node_coords[col] - node_coords[row]
    dist = torch.norm(cart, p=2, dim=-1, keepdim=True)
    return torch.cat([cart, dist], dim=-1)


def predict(simulator: learned_simulator.MeshSimulator,
            device: str):

    # Load simulator
    if os.path.exists(FLAGS.model_path + FLAGS.model_file):
        simulator.load(FLAGS.model_path + FLAGS.model_file)
    else:
        raise Exception(f"Model does not exist at {FLAGS.model_path + FLAGS.model_file}")

    simulator.to(device)
    simulator.eval()

    # Output path
    if not os.path.exists(FLAGS.output_path):
        os.makedirs(FLAGS.output_path)

    # Use `valid`` set for eval mode if not use `test`
    split = 'test' if FLAGS.mode == 'rollout' else 'valid'

    # Load trajectory data.
    ds = data_loader.get_data_loader_by_trajectories(path=f"{FLAGS.data_path}{split}.npz")

    def report(i, prediction_data):
        print(f"Rollout for example{i}: loss = {prediction_data['mean_loss']} {prediction_data['mean_acc_loss']}")
        # Save rollout in testing
        if FLAGS.mode == 'rollout':
            filename = f'{FLAGS.rollout_filename}_{i}.pkl'
            filename = os.path.join(FLAGS.output_path, filename)
            with open(filename, 'wb') as f:
                pickle.dump(prediction_data, f)

    # Rollout
    with torch.no_grad():
        if FLAGS.rollout_batch_size <= 1 and FLAGS.rollout_fast == 'off':
            for i, features in enumerate(ds):
                nsteps = len(features[0]) - INPUT_SEQUENCE_LENGTH
                prediction_data = rollout(simulator, features, nsteps, device)
                report(i, prediction_data)
        else:
            # Batch consecutive trajectories of equal length; output order and filenames are unchanged.
            examples = list(ds)
            i = 0
            while i < len(examples):
                group = [examples[i]]
                while (len(group) < FLAGS.rollout_batch_size and i + len(group) < len(examples)
                       and len(examples[i + len(group)][0]) == len(group[0][0])):
                    group.append(examples[i + len(group)])
                nsteps = len(group[0][0]) - INPUT_SEQUENCE_LENGTH
                if FLAGS.rollout_fast == 'off':
                    outputs = rollout_batched(simulator, group, nsteps, device)
                else:
                    if INPUT_SEQUENCE_LENGTH != 1:
                        raise ValueError('--rollout_fast needs INPUT_SEQUENCE_LENGTH == 1')
                    outputs = rollout_fast(simulator, group, nsteps, device, precision=FLAGS.rollout_fast, dt=dt)
                for k, prediction_data in enumerate(outputs):
                    report(i + k, prediction_data)
                i += len(group)

    print(f"Mean loss on rollout prediction: {prediction_data['mean_loss']} {prediction_data['mean_acc_loss']}")

def _prepare_trajectory(features, device):
    """Move one trajectory's features to device and build its (static) graph.

    Node positions don't move during a rollout, so the graph (edges, edge
    features, node type/property) can be built once from the first frame and
    reused for every step; only the velocities change per step.
    """
    node_coords, node_types, node_property, velocities, pressures, cells = (
        f.to(device) for f in features[:6])

    initial_velocities = velocities[:INPUT_SEQUENCE_LENGTH]
    ground_truth_velocities = velocities[INPUT_SEQUENCE_LENGTH:]
    current_velocities = initial_velocities.squeeze()
    nnodes = current_velocities.shape[0]

    example = (
        (node_coords[0], node_types[0], node_property[0], current_velocities,
         pressures[0], cells[0], torch.zeros(nnodes, device=device)),
        ground_truth_velocities[0])
    graph = transformer(datas_to_graph(example, dt=dt, device=device))

    return dict(node_coords=node_coords, node_types=node_types, node_property=node_property,
                initial_velocities=initial_velocities, ground_truth_velocities=ground_truth_velocities,
                current_velocities=current_velocities, nnodes=nnodes, graph=graph)


def _non_kinematic_mask(node_type):
    """Mask of boundary nodes that take ground-truth velocities rather than a prediction."""
    kinematic_mask = torch.logical_or(node_type == NodeType.NORMAL, node_type == NodeType.HIGH_STRESS)
    kinematic_mask = kinematic_mask.squeeze() if kinematic_mask.dim() > 1 else kinematic_mask
    return ~kinematic_mask


def _run_rollout_steps(simulator, nsteps, current_velocities, node_type, node_property,
                       edge_index, edge_attr, ground_truth_velocities, mask):
    """Step the simulator nsteps times, pinning boundary nodes to ground truth each step."""
    predictions = torch.empty((nsteps,) + tuple(current_velocities.shape),
                              dtype=current_velocities.dtype, device=current_velocities.device)
    for step in tqdm(range(nsteps), total=nsteps):
        predicted_next_velocity = simulator.predict_velocity(
            current_velocities=current_velocities,
            node_type=node_type,
            node_property=node_property,
            edge_index=edge_index,
            edge_features=edge_attr)

        # Apply ground truth velocities at boundary nodes
        predicted_next_velocity[mask] = ground_truth_velocities[step][mask]
        predictions[step] = predicted_next_velocity

        # Update current position for the next prediction
        current_velocities = predicted_next_velocity
    return predictions


def _rollout_output(initial_velocities, predictions, ground_truth_velocities,
                    node_coords, node_types, node_property):
    # Prediction with shape (time, nnodes, dim)
    loss = (predictions - ground_truth_velocities) ** 2
    return {
        'initial_velocities': initial_velocities.cpu().numpy(),
        'predicted_rollout': predictions.cpu().numpy(),
        'ground_truth_rollout': ground_truth_velocities.cpu().numpy(),
        'node_coords': node_coords.cpu().numpy(),
        'node_types': node_types.cpu().numpy(),
        'node_property': node_property.cpu().numpy(),
        'mean_loss': loss.mean().cpu().numpy(),
        'mean_acc_loss': None
    }


def rollout(simulator: learned_simulator.MeshSimulator,
            features,
            nsteps: int,
            device):

    traj = _prepare_trajectory(features, device)
    graph = traj['graph']

    # Cache everything that doesn't change
    cached_edge_index = graph.edge_index
    cached_edge_attr = graph.edge_attr  # Positions don't move!
    cached_node_type = graph.x[:, 0]
    cached_node_property = graph.x[:, 1]

    # Boundary nodes take ground-truth velocities; the mask is static, so compute it once.
    mask = _non_kinematic_mask(cached_node_type)

    predictions = _run_rollout_steps(
        simulator, nsteps, traj['current_velocities'], cached_node_type, cached_node_property,
        cached_edge_index, cached_edge_attr, traj['ground_truth_velocities'], mask)

    return _rollout_output(traj['initial_velocities'], predictions, traj['ground_truth_velocities'],
                           traj['node_coords'], traj['node_types'], traj['node_property'])

def rollout_batched(simulator: learned_simulator.MeshSimulator,
                    features_list,
                    nsteps: int,
                    device):
    """Roll out several same-length trajectories as one disjoint graph.

    Per trajectory the computation is that of rollout(): the graphs share no edges, so
    message passing never mixes trajectories. Returns one output dict per trajectory.
    """
    parts = [_prepare_trajectory(features, device) for features in features_list]

    offsets = [0]
    for p in parts[:-1]:
        offsets.append(offsets[-1] + p['nnodes'])
    edge_index = torch.cat([p['graph'].edge_index + o for p, o in zip(parts, offsets)], dim=1)
    edge_attr = torch.cat([p['graph'].edge_attr for p in parts], dim=0)
    node_type = torch.cat([p['graph'].x[:, 0] for p in parts], dim=0)
    node_prop = torch.cat([p['graph'].x[:, 1] for p in parts], dim=0)
    truth = torch.cat([p['ground_truth_velocities'] for p in parts], dim=1)
    current_velocities = torch.cat([p['current_velocities'] for p in parts], dim=0)

    mask = _non_kinematic_mask(node_type)

    predictions = _run_rollout_steps(
        simulator, nsteps, current_velocities, node_type, node_prop,
        edge_index, edge_attr, truth, mask)

    outputs = []
    for p, o in zip(parts, offsets):
        pred = predictions[:, o:o + p['nnodes']]
        outputs.append(_rollout_output(p['initial_velocities'], pred, p['ground_truth_velocities'],
                                       p['node_coords'], p['node_types'], p['node_property']))
    return outputs

def acceleration_loss(pred_acc, target_acc, non_kinematic_mask):
    errors = ((pred_acc - target_acc)**2)[non_kinematic_mask]  # only compute errors if node_types is NORMAL or OUTFLOW
    loss = torch.mean(errors)
    return loss

def train(simulator):

    print(f"device = {device}")

    # Initiate training.
    optimizer = torch.optim.Adam(simulator.parameters(), lr=lr_init)
    step = 0
    epoch = 0
    steps_per_epoch = 0

    valid_loss = 0
    epoch_train_loss = 0
    epoch_valid_loss = 0 

    train_loss_hist = []
    valid_loss_hist = []
    epoch_ave_train_loss_hist = []
    valid_loss_at_epoch_hist = []

    # Set model and its path to save, and load model.
    # If model_path does not exist create new directory and begin training.
    model_path = FLAGS.model_path
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    # Opt-in seeding (meshnet/seeding.py). FLAGS.seed=None (default): every generator
    # below stays None, which is the exact legacy unseeded call signature everywhere
    # it is threaded through (data_loader.get_data_loader_by_samples(generator=None),
    # get_velocity_noise(generator=None)) -- default behaviour is unchanged.
    train_generator = None
    valid_generator = None
    noise_generator = None
    if FLAGS.seed is not None:
        seed_streams = seeding.sub_seeds(FLAGS.seed)
        train_generator = seeding.torch_generator(seed_streams["data_train"])
        valid_generator = seeding.torch_generator(seed_streams["data_valid"])
        noise_generator = seeding.torch_generator(seed_streams["noise"])

    # If model_path does exist and model_file and train_state_file exist continue training.
    if FLAGS.model_file is not None:

        if FLAGS.model_file == "latest" and FLAGS.train_state_file == "latest":
            # find the latest model, assumes model and train_state files are in step.
            fnames = glob.glob(f"{model_path}*model*pt")
            max_model_number = 0
            expr = re.compile(".*model-(\d+).pt")
            for fname in fnames:
                model_num = int(expr.search(fname).groups()[0])
                if model_num > max_model_number:
                    max_model_number = model_num
            # reset names to point to the latest.
            FLAGS.model_file = f"model-{max_model_number}.pt"
            FLAGS.train_state_file = f"train_state-{max_model_number}.pt"

        if os.path.exists(model_path + FLAGS.model_file) and os.path.exists(model_path + FLAGS.train_state_file):
            # load model
            simulator.load(model_path + FLAGS.model_file)

            # load train state
            train_state = torch.load(model_path + FLAGS.train_state_file)
            # set optimizer state
            optimizer = torch.optim.Adam(simulator.parameters())
            optimizer.load_state_dict(train_state["optimizer_state"])
            optimizer_to(optimizer, device)
            # set global train state
            step = train_state["global_train_state"].pop("step")
            # Restore RNG streams so a resumed run continues them instead of
            # restarting (meshnet/seeding.py). Only present/restored when this
            # run opts in via --seed; absent for pre-existing train_state files.
            if FLAGS.seed is not None and "rng_state" in train_state:
                seeding.restore(
                    train_state["rng_state"],
                    {"data_train": train_generator, "data_valid": valid_generator,
                     "noise": noise_generator},
                    {})
                print("Resumed RNG streams from train_state.")
        else:
            raise FileNotFoundError(
                f"Specified model_file {model_path + FLAGS.model_file} and train_state_file {model_path + FLAGS.train_state_file} not found.")

    simulator.train()
    simulator.to(device)

    # Load data
    ds = data_loader.get_data_loader_by_samples(path=f'{FLAGS.data_path}/{FLAGS.mode}.npz',
                                                input_length_sequence=INPUT_SEQUENCE_LENGTH,
                                                dt=dt,
                                                batch_size=FLAGS.batch_size,
                                                generator=train_generator)

    ds_valid = data_loader.get_data_loader_by_samples(path=f'{FLAGS.data_path}/valid.npz',
                                                      input_length_sequence=INPUT_SEQUENCE_LENGTH,
                                                      dt=dt,
                                                      batch_size=FLAGS.batch_size,
                                                      generator=valid_generator)
    not_reached_nsteps = True
    try:
        while not_reached_nsteps:
            for i, graph in enumerate(ds):
                steps_per_epoch += 1

                # Represent graph using edge_index and make edge_feature to be using [relative_distance, norm]
                graph = transformer(graph.to(device))

                # Get inputs
                node_types = graph.x[:, 0]
                node_property = graph.x[:, 1]
                current_velocities = graph.x[:, 2:4]
                edge_index = graph.edge_index
                edge_features = graph.edge_attr
                target_velocities = graph.y

                # Get velocity noise
                velocity_noise = get_velocity_noise(graph, noise_std=noise_std, device=device,
                                                     generator=noise_generator)
                #print('before predict_acceleration in train loop')
                #print(node_types, node_types.shape)
                #print(node_property, node_property.shape)
                #print(current_velocities, current_velocities.shape)

                # Predict dynamics
                pred_acc, target_acc = simulator.predict_acceleration(
                    current_velocities=current_velocities,
                    node_type=node_types, node_property=node_property,
                    edge_index=edge_index,
                    edge_features=edge_features,
                    target_velocities=target_velocities,
                    velocity_noise=velocity_noise)
                #print('after predict_acceleration in train loop')

                non_kinematic_mask = torch.logical_or(node_types == NodeType.NORMAL, \
                    node_types == NodeType.HIGH_STRESS)

                # validation 
                if step % loss_report_step == 0:
                    sampled_valid_example = next(iter(ds_valid))
                    valid_loss = validation(simulator, sampled_valid_example, device,
                                             noise_generator=noise_generator)
                    #valid_loss_hist.append(valid_loss)

                loss = acceleration_loss(pred_acc, target_acc, non_kinematic_mask)
                epoch_train_loss += loss
                
                current_vel_mag = torch.norm(current_velocities, dim=-1) #(nnode,)
                target_vel_mag = torch.norm(current_velocities, dim=-1) #(nnode,)
                current_vel_rms = torch.sqrt(torch.mean(current_vel_mag**2))
                target_vel_rms = torch.sqrt(torch.mean(target_vel_mag**2))
                current_vel_max = torch.max(current_vel_mag)
                target_vel_max = torch.max(target_vel_mag)  
              
                # Computes the gradient of loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Update learning rate
                lr_new = lr_init * lr_decay_rate ** (step / lr_decay_steps) + 1e-6
                for param in optimizer.param_groups:
                    param['lr'] = lr_new

                if step % loss_report_step == 0:
                    print(f"Training step: {step}/{FLAGS.ntraining_steps}. Train loss: {loss}. Valid loss: {valid_loss}")
                    print(f"RMS Current vel {current_vel_rms}. Max current vel {current_vel_max}")
                    with open(model_path + 'loss_log.txt', 'a') as f:
                        f.write(f"{step} {loss} {valid_loss} {current_vel_rms} {target_vel_rms} {current_vel_max} {target_vel_max}\n")

                # Save model state
                if step % FLAGS.nsave_steps == 0:
                    simulator.save(model_path + 'model-' + str(step) + '.pt')
                    # Resume off-by-one fix: this step's optimizer.step() has already been
                    # applied above (line ~303) by the time we get here, so the state we are
                    # about to save reflects "step" fully processed. Record step+1 as the
                    # resume point, so `step = train_state["global_train_state"].pop("step")`
                    # re-enters the loop at the FIRST UNPROCESSED step, not at "step" again
                    # (the pre-fix code saved plain `step`, which caused resume to redo this
                    # step's gradient update a second time and then immediately overwrite
                    # this very model-<step>.pt/train_state-<step>.pt with the doubly-updated
                    # state). The saved checkpoint FILENAME is left as `step` (unchanged) --
                    # only the resume-point integer inside train_state changes.
                    train_state = dict(optimizer_state=optimizer.state_dict(),
                                        global_train_state={"step": step + 1})
                    if FLAGS.seed is not None:
                        train_state["seed"] = FLAGS.seed
                        train_state["rng_state"] = seeding.capture(
                            {"data_train": train_generator, "data_valid": valid_generator,
                             "noise": noise_generator},
                            {})
                    torch.save(train_state, f"{model_path}train_state-{step}.pt")

                # Complete training
                if (step >= FLAGS.ntraining_steps):
                    not_reached_nsteps = False
                    break

                step += 1

            # Epoch level statistics
            # Record training loss at epoch
            epoch_train_loss /= steps_per_epoch
            #epoch_ave_train_loss_hist.append(epoch_train_loss)
            #valid_loss_at_epoch_hist.append(valid_loss)
            with open(model_path + 'epoch_loss_log.txt', 'a') as f:
                f.write(f"{step} {epoch} {epoch_train_loss} {valid_loss}\n")
            print('')
            print(f"At epoch {epoch} and step {step}, epoch train loss: {epoch_train_loss}; valid loss: {valid_loss}")
            if steps_per_epoch >= len(graph):
                epoch += 1
                
            steps_per_epoch = 0
            epoch_train_loss = 0

    except KeyboardInterrupt:
        pass

def validation(
    simulator,
    graph,
    device,
    noise_generator=None
    ):
    graph = transformer(graph.to(device))
    node_types = graph.x[:, 0]  # (nnodes, )
    node_property = graph.x[:, 1]  # (nnodes, )
    current_velocities = graph.x[:, 2:4]  # (nnodes, 2)
    edge_index = graph.edge_index  # (2, nedges)
    edge_features = graph.edge_attr  # (nedges, 2)
    target_velocities = graph.y  # (nnodes, 2)

    # Get velocity noise
    velocity_noise = get_velocity_noise(graph, noise_std=noise_std, device=device,
                                         generator=noise_generator)
    pred_acc, target_acc = simulator.predict_acceleration(
        current_velocities=current_velocities,
        node_type=node_types,
        node_property=node_property,
        edge_index=edge_index,
        edge_features=edge_features,
        target_velocities=target_velocities,
        velocity_noise=velocity_noise)

    non_kinematic_mask = torch.logical_or(node_types == NodeType.NORMAL, \
                    node_types == NodeType.HIGH_STRESS)

    loss = acceleration_loss(pred_acc, target_acc, non_kinematic_mask)
    return loss

def main(_):
    global config 
    global INPUT_SEQUENCE_LENGTH
    global noise_std
    global node_type_embedding_size
    global dt
    global lr_init
    global lr_decay_rate
    global lr_decay_steps
    global loss_report_step

    # default system and GNS simulator parameters
    config_file_path = f"{FLAGS.model_path}/config.json"
    if os.path.exists(config_file_path):
        with open(config_file_path, 'r') as f:
            config = json.load(f)
            print('config file found.')
            print('System and GNS simulator configuration is', config)

            INPUT_SEQUENCE_LENGTH = config['INPUT_SEQUENCE_LENGTH']
            noise_std = config['noise_std']
            node_type_embedding_size = config['node_type_embedding_size']
            dt = config['dt']
            lr_init = config['lr_init']
            lr_decay_rate = config['lr_decay_rate']
            lr_decay_steps = config['lr_decay_steps']
            loss_report_step = config['loss_report_step']
            simulator_simulation_dimensions = config['simulator_simulation_dimensions']
            simulator_nnode_in = config['simulator_nnode_in']
            simulator_nedge_in = config['simulator_nedge_in']
            simulator_latent_dim = config['simulator_latent_dim']
            simulator_nmessage_passing_steps = config['simulator_nmessage_passing_steps']
            simulator_nmlp_layers = config['simulator_nmlp_layers']
            simulator_mlp_hidden_dim = config['simulator_mlp_hidden_dim']
            simulator_nnode_types = config['simulator_nnode_types']
            simulator_node_type_embedding_size = config['simulator_node_type_embedding_size']
    else:
        print('No config file found. Using default parameters.')
        INPUT_SEQUENCE_LENGTH = 1
        noise_std = 2e-2
        node_type_embedding_size = 9
        dt=0.041666666666667
        lr_init = 1e-4
        lr_decay_rate = 0.1
        lr_decay_steps = 5e6
        loss_report_step = 1000
        simulator_simulation_dimensions=2
        simulator_nnode_in=12
        simulator_nedge_in=3
        simulator_latent_dim=128
        simulator_nmessage_passing_steps=15
        simulator_nmlp_layers=2
        simulator_mlp_hidden_dim=128
        simulator_nnode_types=3
        simulator_node_type_embedding_size=9

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if FLAGS.cuda_device_number is not None and torch.cuda.is_available():
        device = torch.device(f'cuda:{int(FLAGS.cuda_device_number)}')

    # Opt-in seeding (meshnet/seeding.py): default FLAGS.seed=None leaves model init
    # exactly as unseeded as before. When set, seed the "init" sub-stream right before
    # building the model so its weight initialization is fixed by the seed.
    if FLAGS.seed is not None:
        seeding.set_deterministic(FLAGS.deterministic)
        seeding.seed_global(seeding.sub_seeds(FLAGS.seed)["init"])

    # load simulator
    simulator = learned_simulator.MeshSimulator(
        simulation_dimensions  = simulator_simulation_dimensions,
        nnode_in               = simulator_nnode_in, 
        nedge_in               = simulator_nedge_in,
        latent_dim             = simulator_latent_dim,
        nmessage_passing_steps = simulator_nmessage_passing_steps,
        nmlp_layers            = simulator_nmlp_layers,
        mlp_hidden_dim         = simulator_mlp_hidden_dim,
        nnode_types            = simulator_nnode_types,
        node_type_embedding_size = simulator_node_type_embedding_size,
        device=device)

    if FLAGS.mode == 'train':
        train(simulator)
    elif FLAGS.mode in ['valid', 'rollout']:
        predict(simulator, device)

if __name__ == "__main__":
    app.run(main)
