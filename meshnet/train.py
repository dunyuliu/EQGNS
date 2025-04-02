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
FLAGS = flags.FLAGS

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# an instance that transforms face-based graph to edge-based graph. Edge features are auto-computed using "Cartesian" and "Distance"
transformer = T.Compose([T.FaceToEdge(), T.Cartesian(norm=False), T.Distance(norm=False)])


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

    # Rollout
    with torch.no_grad():
        for i, features in enumerate(ds):
            nsteps = len(features[0]) - INPUT_SEQUENCE_LENGTH
            prediction_data = rollout(simulator, features, nsteps, device)
            print(f"Rollout for example{i}: loss = {prediction_data['mean_loss']} {prediction_data['mean_acc_loss']}")

            # Save rollout in testing
            if FLAGS.mode == 'rollout':
                filename = f'{FLAGS.rollout_filename}_{i}.pkl'
                filename = os.path.join(FLAGS.output_path, filename)
                with open(filename, 'wb') as f:
                    pickle.dump(prediction_data, f)

    print(f"Mean loss on rollout prediction: {prediction_data['mean_loss']} {prediction_data['mean_acc_loss']}")

def rollout(simulator: learned_simulator.MeshSimulator,
            features,
            nsteps: int,
            device):

    node_coords = features[0]  # (timesteps, nnode, ndims)
    node_types = features[1]  # (timesteps, nnode, )
    node_property = features[2]  # (timesteps, nnode, )
    velocities = features[3]  # (timesteps, nnode, ndims)
    pressures = features[4]  # (timesteps, nnode, )
    cells = features[5]  # # (timesteps, ncells, nnode_per_cell)

    initial_velocities = velocities[:INPUT_SEQUENCE_LENGTH]
    ground_truth_velocities = velocities[INPUT_SEQUENCE_LENGTH:]

    current_velocities = initial_velocities.squeeze().to(device)
    predictions = []
    acc_loss = []

    mask = None

    for step in tqdm(range(nsteps), total=nsteps):

        # Predict next velocity
        # First, obtain data to form a graph
        current_node_coords = node_coords[step]
        current_node_type = node_types[step]
        current_node_property = node_property[step]
        current_pressure = pressures[step]
        current_cell = cells[step]
        current_time_idx_vector = torch.tensor(np.full(current_node_coords.shape[0], step)).to(torch.float32).contiguous()
        next_ground_truth_velocities = ground_truth_velocities[step].to(device)
        current_example = (
            (current_node_coords, current_node_type, current_node_property, current_velocities, current_pressure, current_cell, current_time_idx_vector),
            next_ground_truth_velocities)

        # Make graph
        graph = datas_to_graph(current_example, dt=dt, device=device)
        # Represent graph using edge_index and make edge_feature to be using [relative_distance, norm]
        graph = transformer(graph)

        # Predict next velocity
        predicted_next_velocity = simulator.predict_velocity(
            current_velocities=graph.x[:, 2:4],
            node_type=graph.x[:, 0],
            node_property=graph.x[:, 1],
            edge_index=graph.edge_index,
            edge_features=graph.edge_attr)
        
        # Get velocity noise
        velocity_noise = get_velocity_noise(graph, noise_std=0.0, device=device)

        # Predict dynamics
        pred_acc, target_acc = simulator.predict_acceleration(
            current_velocities=graph.x[:,2:4],
            node_type=graph.x[:,0],
            node_property=graph.x[:,1],
            edge_index=graph.edge_index,
            edge_features=graph.edge_attr,
            target_velocities=graph.y,
            velocity_noise=velocity_noise)

        # Apply mask.
        if mask is None:  # only compute mask for the first timestep, since it will be the same for the later timesteps
            mask0 = torch.logical_or(current_node_type == NodeType.NORMAL, \
					current_node_type == NodeType.HIGH_STRESS)
            mask = mask0
            mask0 = mask0.squeeze(1)
            mask = torch.logical_not(mask)
            mask = mask.squeeze(1)
        # Maintain previous velocity if node_type is not (Normal or Outflow).
        # i.e., only update normal or outflow nodes.
        predicted_next_velocity[mask] = next_ground_truth_velocities[mask]
        predictions.append(predicted_next_velocity)
        
        errors = ((pred_acc-target_acc)**2)[mask0]
        acc_loss.append(torch.mean(errors))
        # Update current position for the next prediction
        current_velocities = predicted_next_velocity.to(device)

    # Prediction with shape (time, nnodes, dim)
    predictions = torch.stack(predictions)
    loss = (predictions - ground_truth_velocities.to(device)) ** 2
    
    acc_loss = torch.stack(acc_loss) 
    loss1 = acc_loss.mean()

    output_dict = {
        'initial_velocities': initial_velocities.cpu().numpy(),
        'predicted_rollout': predictions.cpu().numpy(),
        'ground_truth_rollout': ground_truth_velocities.cpu().numpy(),
        'node_coords': node_coords.cpu().numpy(),
        'node_types': node_types.cpu().numpy(),
        'node_property': node_property.cpu().numpy(),
        'mean_loss': loss.mean().cpu().numpy(),
        'mean_acc_loss':loss1.cpu().numpy()
    }

    return output_dict

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
        else:
            raise FileNotFoundError(
                f"Specified model_file {model_path + FLAGS.model_file} and train_state_file {model_path + FLAGS.train_state_file} not found.")

    simulator.train()
    simulator.to(device)

    # Load data
    ds = data_loader.get_data_loader_by_samples(path=f'{FLAGS.data_path}/{FLAGS.mode}.npz',
                                                input_length_sequence=INPUT_SEQUENCE_LENGTH,
                                                dt=dt,
                                                batch_size=FLAGS.batch_size)

    ds_valid = data_loader.get_data_loader_by_samples(path=f'{FLAGS.data_path}/valid.npz',
                                                      input_length_sequence=INPUT_SEQUENCE_LENGTH,
                                                      dt=dt,
                                                      batch_size=FLAGS.batch_size)
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
                velocity_noise = get_velocity_noise(graph, noise_std=noise_std, device=device)
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
                    valid_loss = validation(simulator, sampled_valid_example, device)
                    #valid_loss_hist.append(valid_loss)

                loss = acceleration_loss(pred_acc, target_acc, non_kinematic_mask)
                epoch_train_loss += loss
                #train_loss_hist.append(loss)

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
                    with open(model_path + 'loss_log.txt', 'a') as f:
                        f.write(f"{step} {loss} {valid_loss}\n")

                # Save model state
                if step % FLAGS.nsave_steps == 0:
                    simulator.save(model_path + 'model-' + str(step) + '.pt')
                    train_state = dict(optimizer_state=optimizer.state_dict(), global_train_state={"step": step})
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
    device
    ):
    graph = transformer(graph.to(device))
    node_types = graph.x[:, 0]  # (nnodes, )
    node_property = graph.x[:, 1]  # (nnodes, )
    current_velocities = graph.x[:, 2:4]  # (nnodes, 2)
    edge_index = graph.edge_index  # (2, nedges)
    edge_features = graph.edge_attr  # (nedges, 2)
    target_velocities = graph.y  # (nnodes, 2)

    # Get velocity noise
    velocity_noise = get_velocity_noise(graph, noise_std=noise_std, device=device)  
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
