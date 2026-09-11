import sys
import os
import glob
import numpy as np
import torch
import torch_geometric.transforms as T
import re
import pickle
from tqdm import tqdm
import json
from typing import List, Tuple

from absl import flags
from absl import app

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'meshnet')))
from meshnet import data_loader
from meshnet import learned_simulator
from meshnet.noise import get_velocity_noise
from meshnet.utils import datas_to_graph
from meshnet.utils import NodeType
from meshnet.utils import optimizer_to

flags.DEFINE_enum(
    'mode', 'rollout', ['rollout'],
    help='Rollout evaluation mode.')
flags.DEFINE_integer('batch_size', 4, help='The number of .pkl files to process in each batch.')
flags.DEFINE_string('working_dir', None, help='The working directory containing dataset/ and models.{model_suffix}/')
flags.DEFINE_string('model_suffix', None, help='Model suffix for finding the model path.')
flags.DEFINE_string('model_ids', None, help='Comma-separated model IDs for rollout.')
flags.DEFINE_integer('gpu_id', 0, help='GPU ID to use.')
flags.DEFINE_string('data_path', None, help='The dataset directory containing .npz files (optional, will use working_dir/dataset/ if not specified).')
flags.DEFINE_string('pkl_path', None, help='The directory containing .pkl files to process in batch.')
flags.DEFINE_string('model_path', None, help=('The path for saving checkpoints of the model (optional, will use working_dir/models.{model_suffix}/ if not specified).'))
flags.DEFINE_string('output_path', None, help='The path for saving outputs (optional, will use working_dir/rollouts.{model_suffix}/{model_file}/ if not specified).')
flags.DEFINE_string('model_file', None, help=('Model filename (.pt) to resume from. Can also use "latest" to default to newest file.'))
flags.DEFINE_integer("cuda_device_number", None, help="CUDA device (zero indexed), default is None so default CUDA device will be used.")
flags.DEFINE_string('rollout_filename', "rollout", help='Name saving the rollout')
FLAGS = flags.FLAGS

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
transformer = T.Compose([T.FaceToEdge(), T.Cartesian(norm=False), T.Distance(norm=False)])

def load_pkl_files(pkl_directory: str) -> List[str]:
    """Load all .pkl files from the specified directory"""
    pkl_files = glob.glob(os.path.join(pkl_directory, "*.pkl"))
    pkl_files.sort()  # Sort for consistent processing order
    return pkl_files

def load_pkl_batch(pkl_files: List[str]) -> List:
    """Load a batch of .pkl files and return their data"""
    batch_data = []
    for pkl_file in pkl_files:
        try:
            with open(pkl_file, 'rb') as f:
                data = pickle.load(f)
                batch_data.append((pkl_file, data))
                print(f"Loaded: {os.path.basename(pkl_file)}")
        except Exception as e:
            print(f"Error loading {pkl_file}: {e}")
            continue
    return batch_data

def batch_predict(simulator: learned_simulator.MeshSimulator,
                  device: str):
    """Batch rollout prediction function for .pkl files or .npz files"""
    
    # Load simulator
    if FLAGS.model_file == "latest":
        # Find the latest model
        fnames = glob.glob(f"{FLAGS.model_path}*model*pt")
        max_model_number = 0
        expr = re.compile(".*model-(\d+).pt")
        for fname in fnames:
            model_num = int(expr.search(fname).groups()[0])
            if model_num > max_model_number:
                max_model_number = model_num
        FLAGS.model_file = f"model-{max_model_number}.pt"
    
    model_file_path = os.path.join(FLAGS.model_path, FLAGS.model_file)
    if os.path.exists(model_file_path):
        simulator.load(model_file_path)
    else:
        raise Exception(f"Model does not exist at {model_file_path}")

    simulator.to(device)
    simulator.eval()

    # Output path
    if not os.path.exists(FLAGS.output_path):
        os.makedirs(FLAGS.output_path)

    if FLAGS.pkl_path:
        # Process .pkl files in batches
        print(f"Processing .pkl files from: {FLAGS.pkl_path}")
        pkl_files = load_pkl_files(FLAGS.pkl_path)
        
        if not pkl_files:
            print(f"No .pkl files found in {FLAGS.pkl_path}")
            return
            
        print(f"Found {len(pkl_files)} .pkl files")
        
        # Process files in batches
        total_batches = (len(pkl_files) + FLAGS.batch_size - 1) // FLAGS.batch_size
        
        for batch_idx in range(total_batches):
            start_idx = batch_idx * FLAGS.batch_size
            end_idx = min((batch_idx + 1) * FLAGS.batch_size, len(pkl_files))
            batch_files = pkl_files[start_idx:end_idx]
            
            print(f"\nProcessing batch {batch_idx + 1}/{total_batches} ({len(batch_files)} files)")
            
            # Load batch data
            batch_data = load_pkl_batch(batch_files)
            
            if not batch_data:
                print(f"No valid data in batch {batch_idx + 1}")
                continue
                
            # Process files in batch simultaneously
            with torch.no_grad():
                try:
                    # Prepare batch data for parallel processing
                    batch_features = []
                    valid_files = []

                    for pkl_file, data in batch_data:
                        try:
                            features = pkl_to_features(data)
                            nsteps = len(features[0]) - INPUT_SEQUENCE_LENGTH

                            if nsteps <= 0:
                                print(f"Skipping {os.path.basename(pkl_file)}: insufficient timesteps")
                                continue

                            batch_features.append((features, nsteps))
                            valid_files.append(pkl_file)

                        except Exception as e:
                            print(f"Error preparing {os.path.basename(pkl_file)}: {e}")
                            continue

                    if not batch_features:
                        print("No valid files in batch")
                        continue

                    # Run batch rollout for all valid files simultaneously
                    batch_predictions = batch_rollout_parallel(simulator, batch_features, device)

                    # Save results for each file
                    for i, (pkl_file, prediction_data) in enumerate(zip(valid_files, batch_predictions)):
                        print(f"Rollout for {os.path.basename(pkl_file)}: loss = {prediction_data['mean_loss']:.6f} acc_loss = {prediction_data['mean_acc_loss']:.6f}")

                        # Save rollout with original filename suffix
                        base_name = os.path.splitext(os.path.basename(pkl_file))[0]
                        output_filename = f'{FLAGS.rollout_filename}_{base_name}.pkl'
                        output_path = os.path.join(FLAGS.output_path, output_filename)

                        with open(output_path, 'wb') as f:
                            pickle.dump(prediction_data, f)

                except Exception as e:
                    print(f"Error in batch processing: {e}")
                    # Fallback to sequential processing
                    print("Falling back to sequential processing...")
                    for pkl_file, data in batch_data:
                        try:
                            features = pkl_to_features(data)
                            nsteps = len(features[0]) - INPUT_SEQUENCE_LENGTH

                            if nsteps <= 0:
                                print(f"Skipping {os.path.basename(pkl_file)}: insufficient timesteps")
                                continue

                            prediction_data = rollout(simulator, features, nsteps, device)
                            print(f"Rollout for {os.path.basename(pkl_file)}: loss = {prediction_data['mean_loss']:.6f} acc_loss = {prediction_data['mean_acc_loss']:.6f}")

                            # Save rollout with original filename suffix
                            base_name = os.path.splitext(os.path.basename(pkl_file))[0]
                            output_filename = f'{FLAGS.rollout_filename}_{base_name}.pkl'
                            output_path = os.path.join(FLAGS.output_path, output_filename)

                            with open(output_path, 'wb') as f:
                                pickle.dump(prediction_data, f)

                        except Exception as e:
                            print(f"Error processing {os.path.basename(pkl_file)}: {e}")
                            continue
                        
    else:
        # Process .npz files (original functionality)
        print(f"Processing .npz files from: {FLAGS.data_path}")
        split = 'test'
        ds = data_loader.get_data_loader_by_trajectories(path=os.path.join(FLAGS.data_path, f"{split}.npz"))

        # Batch rollout
        with torch.no_grad():
            for i, features in enumerate(ds):
                nsteps = len(features[0]) - INPUT_SEQUENCE_LENGTH
                prediction_data = rollout(simulator, features, nsteps, device)
                print(f"Rollout for example{i}: loss = {prediction_data['mean_loss']:.6f} acc_loss = {prediction_data['mean_acc_loss']:.6f}")

                # Save rollout
                filename = f'{FLAGS.rollout_filename}_{i}.pkl'
                filename = os.path.join(FLAGS.output_path, filename)
                with open(filename, 'wb') as f:
                    pickle.dump(prediction_data, f)

    print(f"Batch rollout completed. Results saved to {FLAGS.output_path}")

def pkl_to_features(pkl_data):
    """Convert pkl data to features format expected by rollout function"""
    # Extract trajectory data from pkl file
    # Assuming pkl contains similar structure to rollout output
    if 'node_coords' in pkl_data:
        node_coords = torch.tensor(pkl_data['node_coords']).to(torch.float32)
        node_types = torch.tensor(pkl_data['node_types'])
        node_property = torch.tensor(pkl_data['node_property']).to(torch.float32)
        
        # Reconstruct velocities from initial and predicted rollout
        if 'initial_velocities' in pkl_data and 'predicted_rollout' in pkl_data:
            initial_velocities = torch.tensor(pkl_data['initial_velocities']).to(torch.float32)
            predicted_rollout = torch.tensor(pkl_data['predicted_rollout']).to(torch.float32)
            # Combine initial and predicted velocities
            velocities = torch.cat([initial_velocities, predicted_rollout], dim=0)
        elif 'ground_truth_rollout' in pkl_data:
            # Use ground truth if available
            initial_velocities = torch.tensor(pkl_data['initial_velocities']).to(torch.float32)
            ground_truth = torch.tensor(pkl_data['ground_truth_rollout']).to(torch.float32)
            velocities = torch.cat([initial_velocities, ground_truth], dim=0)
        else:
            raise ValueError("No velocity data found in pkl file")
            
        # Create dummy pressure and cells data if not available
        timesteps, nnodes, ndims = node_coords.shape
        pressures = torch.zeros(timesteps, nnodes, 1)
        cells = torch.zeros(timesteps, 1, 3)  # Dummy cells data
        
        features = (node_coords, node_types, node_property, velocities, pressures, cells)
        return features
    else:
        raise ValueError("Unsupported pkl file format: missing required fields")

def rollout(simulator: learned_simulator.MeshSimulator,
            features,
            nsteps: int,
            device):
    """Rollout function for trajectory prediction"""
    
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

def batch_rollout_parallel(simulator: learned_simulator.MeshSimulator,
                          batch_features: List[Tuple],
                          device):
    """Parallel rollout function for multiple scenarios"""

    if len(batch_features) == 1:
        # Single scenario - use regular rollout
        features, nsteps = batch_features[0]
        return [rollout(simulator, features, nsteps, device)]

    # Multiple scenarios - process in parallel batches
    batch_results = []

    for features, nsteps in batch_features:
        # For now, process each scenario individually but in the same batch
        # Future enhancement: true parallel processing with batched tensor operations
        result = rollout(simulator, features, nsteps, device)
        batch_results.append(result)

    return batch_results

def batch_rollout_parallel_optimized(simulator: learned_simulator.MeshSimulator,
                                   batch_features: List[Tuple],
                                   device):
    """Optimized parallel rollout with batched tensor operations"""

    if len(batch_features) == 1:
        features, nsteps = batch_features[0]
        return [rollout(simulator, features, nsteps, device)]

    # Check if all scenarios have the same dimensions and timesteps
    first_features, first_nsteps = batch_features[0]
    can_batch = True

    for features, nsteps in batch_features[1:]:
        if (features[0].shape != first_features[0].shape or
            features[1].shape != first_features[1].shape or
            nsteps != first_nsteps):
            can_batch = False
            break

    if not can_batch:
        # Fall back to sequential processing if scenarios have different shapes
        print("Scenarios have different shapes, falling back to sequential processing")
        return batch_rollout_parallel(simulator, batch_features, device)

    # True batched processing for scenarios with same dimensions
    print(f"Processing {len(batch_features)} scenarios in parallel batch")

    # Stack all scenarios into batch tensors
    batch_node_coords = torch.stack([features[0] for features, _ in batch_features])
    batch_node_types = torch.stack([features[1] for features, _ in batch_features])
    batch_node_property = torch.stack([features[2] for features, _ in batch_features])
    batch_velocities = torch.stack([features[3] for features, _ in batch_features])
    batch_pressures = torch.stack([features[4] for features, _ in batch_features])
    batch_cells = torch.stack([features[5] for features, _ in batch_features])

    nsteps = first_nsteps
    batch_size = len(batch_features)

    # Split into initial and ground truth
    batch_initial_velocities = batch_velocities[:, :INPUT_SEQUENCE_LENGTH]
    batch_ground_truth_velocities = batch_velocities[:, INPUT_SEQUENCE_LENGTH:]

    batch_current_velocities = batch_initial_velocities.squeeze().to(device)
    batch_predictions = []
    batch_acc_loss = []

    masks = None

    for step in tqdm(range(nsteps), total=nsteps, desc=f"Batch rollout (x{batch_size})"):
        step_predictions = []
        step_acc_losses = []

        # Process each scenario in the batch
        for b in range(batch_size):
            current_node_coords = batch_node_coords[b, step]
            current_node_type = batch_node_types[b, step]
            current_node_property = batch_node_property[b, step]
            current_pressure = batch_pressures[b, step]
            current_cell = batch_cells[b, step]
            current_velocities = batch_current_velocities[b]

            current_time_idx_vector = torch.tensor(np.full(current_node_coords.shape[0], step)).to(torch.float32).contiguous()
            next_ground_truth_velocities = batch_ground_truth_velocities[b, step].to(device)

            current_example = (
                (current_node_coords, current_node_type, current_node_property, current_velocities, current_pressure, current_cell, current_time_idx_vector),
                next_ground_truth_velocities)

            # Make graph
            graph = datas_to_graph(current_example, dt=dt, device=device)
            graph = transformer(graph)

            # Predict velocity and acceleration
            predicted_next_velocity = simulator.predict_velocity(
                current_velocities=graph.x[:, 2:4],
                node_type=graph.x[:, 0],
                node_property=graph.x[:, 1],
                edge_index=graph.edge_index,
                edge_features=graph.edge_attr)

            velocity_noise = get_velocity_noise(graph, noise_std=0.0, device=device)

            pred_acc, target_acc = simulator.predict_acceleration(
                current_velocities=graph.x[:,2:4],
                node_type=graph.x[:,0],
                node_property=graph.x[:,1],
                edge_index=graph.edge_index,
                edge_features=graph.edge_attr,
                target_velocities=graph.y,
                velocity_noise=velocity_noise)

            # Apply mask
            if masks is None:
                masks = []
                for bb in range(batch_size):
                    mask0 = torch.logical_or(batch_node_types[bb, step] == NodeType.NORMAL,
                                           batch_node_types[bb, step] == NodeType.HIGH_STRESS)
                    mask = torch.logical_not(mask0.squeeze(1))
                    masks.append((mask0.squeeze(1), mask))

            mask0, mask = masks[b]
            predicted_next_velocity[mask] = next_ground_truth_velocities[mask]
            step_predictions.append(predicted_next_velocity)

            errors = ((pred_acc-target_acc)**2)[mask0]
            step_acc_losses.append(torch.mean(errors))

            # Update current velocities for next step
            batch_current_velocities[b] = predicted_next_velocity.to(device)

        batch_predictions.append(torch.stack(step_predictions))
        batch_acc_loss.append(torch.stack(step_acc_losses))

    # Stack predictions and compute losses
    batch_predictions = torch.stack(batch_predictions, dim=1)  # (batch_size, time, nnodes, dim)
    batch_acc_loss = torch.stack(batch_acc_loss, dim=1)  # (batch_size, time)

    # Compute individual scenario results
    results = []
    for b in range(batch_size):
        predictions = batch_predictions[b]
        ground_truth = batch_ground_truth_velocities[b].to(device)
        loss = (predictions - ground_truth) ** 2
        acc_loss = batch_acc_loss[b].mean()

        features, _ = batch_features[b]
        output_dict = {
            'initial_velocities': batch_initial_velocities[b].cpu().numpy(),
            'predicted_rollout': predictions.cpu().numpy(),
            'ground_truth_rollout': ground_truth.cpu().numpy(),
            'node_coords': features[0].cpu().numpy(),
            'node_types': features[1].cpu().numpy(),
            'node_property': features[2].cpu().numpy(),
            'mean_loss': loss.mean().cpu().numpy(),
            'mean_acc_loss': acc_loss.cpu().numpy()
        }
        results.append(output_dict)

    return results

def main(_):
    global INPUT_SEQUENCE_LENGTH
    global noise_std
    global node_type_embedding_size
    global dt
    
    # Set up paths based on working_dir structure (similar to run.process.gns.py)
    if FLAGS.working_dir and FLAGS.model_suffix:
        # Use working_dir structure
        if not FLAGS.model_path:
            FLAGS.model_path = os.path.join(FLAGS.working_dir, f'models.{FLAGS.model_suffix}')
        if not FLAGS.data_path and not FLAGS.pkl_path:
            FLAGS.data_path = os.path.join(FLAGS.working_dir, 'dataset')
    
    # Process model_ids if provided
    if FLAGS.model_ids:
        model_ids = [int(mid.strip()) for mid in FLAGS.model_ids.split(',')]
    else:
        model_ids = [None]  # Process once without specific model ID
    
    # Validate input arguments
    if not FLAGS.pkl_path and not FLAGS.data_path:
        raise ValueError("Either --pkl_path or --data_path must be specified")
    if not FLAGS.model_path:
        raise ValueError("--model_path must be specified")
        
    print(f"Batch rollout script starting...")
    print(f"Working directory: {FLAGS.working_dir}")
    print(f"Model suffix: {FLAGS.model_suffix}")
    print(f"Model path: {FLAGS.model_path}")
    print(f"Batch size: {FLAGS.batch_size}")
    if FLAGS.pkl_path:
        print(f"Processing .pkl files from: {FLAGS.pkl_path}")
    else:
        print(f"Processing .npz files from: {FLAGS.data_path}")
    
    # Load configuration from config.json
    config_file_path = f"{FLAGS.model_path}/config.json"
    if os.path.exists(config_file_path):
        with open(config_file_path, 'r') as f:
            config = json.load(f)
            print('Config file found.')
            print('System and GNS simulator configuration is', config)

            INPUT_SEQUENCE_LENGTH = config['INPUT_SEQUENCE_LENGTH']
            noise_std = config['noise_std']
            node_type_embedding_size = config['node_type_embedding_size']
            dt = config['dt']
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
        dt = 0.041666666666667
        simulator_simulation_dimensions = 2
        simulator_nnode_in = 12
        simulator_nedge_in = 3
        simulator_latent_dim = 128
        simulator_nmessage_passing_steps = 15
        simulator_nmlp_layers = 2
        simulator_mlp_hidden_dim = 128
        simulator_nnode_types = 3
        simulator_node_type_embedding_size = 9

    # Set device - use gpu_id if cuda_device_number not specified
    if FLAGS.cuda_device_number is not None and torch.cuda.is_available():
        device = torch.device(f'cuda:{int(FLAGS.cuda_device_number)}')
    elif FLAGS.gpu_id is not None and torch.cuda.is_available():
        device = torch.device(f'cuda:{int(FLAGS.gpu_id)}')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"Using device: {device}")

    # Load simulator
    simulator = learned_simulator.MeshSimulator(
        simulation_dimensions=simulator_simulation_dimensions,
        nnode_in=simulator_nnode_in, 
        nedge_in=simulator_nedge_in,
        latent_dim=simulator_latent_dim,
        nmessage_passing_steps=simulator_nmessage_passing_steps,
        nmlp_layers=simulator_nmlp_layers,
        mlp_hidden_dim=simulator_mlp_hidden_dim,
        nnode_types=simulator_nnode_types,
        node_type_embedding_size=simulator_node_type_embedding_size,
        device=device)

    # Handle multiple model suffixes (comma-separated)
    model_suffixes = FLAGS.model_suffix.split(',') if ',' in FLAGS.model_suffix else [FLAGS.model_suffix]

    print(f"Processing {len(model_suffixes)} model suffixes: {model_suffixes}")

    # Process each model suffix and model_id combination
    for model_suffix in model_suffixes:
        model_suffix = model_suffix.strip()  # Remove any whitespace

        # Update model path for current suffix
        if FLAGS.working_dir and model_suffix:
            FLAGS.model_path = os.path.join(FLAGS.working_dir, f'models.{model_suffix}')
            print(f"Model path for {model_suffix}: {FLAGS.model_path}")

        for model_id in model_ids:
            if model_id is not None:
                print(f"\nProcessing model ID: {model_id} with suffix: {model_suffix}")
                FLAGS.model_file = f'model-{model_id}.pt'

                # Set output path for current model suffix
                if FLAGS.working_dir and model_suffix and not FLAGS.output_path:
                    FLAGS.output_path = os.path.join(FLAGS.working_dir, f'rollouts.{model_suffix}', FLAGS.model_file)
                    os.makedirs(FLAGS.output_path, exist_ok=True)
                    print(f"Output path: {FLAGS.output_path}")

            # Load configuration for this specific model
            config_file_path = f"{FLAGS.model_path}/config.json"
            if os.path.exists(config_file_path):
                with open(config_file_path, 'r') as f:
                    model_config = json.load(f)
                    print(f'Loading config for {model_suffix}: {model_config}')

                # Create simulator with model-specific configuration
                model_simulator = learned_simulator.MeshSimulator(
                    simulation_dimensions=model_config.get('simulator_simulation_dimensions', simulator_simulation_dimensions),
                    nnode_in=model_config.get('simulator_nnode_in', simulator_nnode_in),
                    nedge_in=model_config.get('simulator_nedge_in', simulator_nedge_in),
                    latent_dim=model_config.get('simulator_latent_dim', simulator_latent_dim),
                    nmessage_passing_steps=model_config.get('simulator_nmessage_passing_steps', simulator_nmessage_passing_steps),
                    nmlp_layers=model_config.get('simulator_nmlp_layers', simulator_nmlp_layers),
                    mlp_hidden_dim=model_config.get('simulator_mlp_hidden_dim', simulator_mlp_hidden_dim),
                    nnode_types=model_config.get('simulator_nnode_types', simulator_nnode_types),
                    node_type_embedding_size=model_config.get('simulator_node_type_embedding_size', simulator_node_type_embedding_size),
                    device=device)
            else:
                print(f'No config file found for {model_suffix}, using default simulator')
                model_simulator = simulator

            # Run batch rollout for this model
            batch_predict(model_simulator, device)

            # Reset output path for next model
            FLAGS.output_path = None

if __name__ == "__main__":
    app.run(main)