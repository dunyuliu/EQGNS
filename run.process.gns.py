import subprocess
import argparse
import os
from itertools import product

def run_command(command, env):
    """Runs a command and prints its output."""
    # Set environment variables for this process
    original_env = {}
    for key, value in env.items():
        original_env[key] = os.environ.get(key)
        os.environ[key] = value
    
    try:
        return os.system(command)
    finally:
        # Restore original environment
        for key, original_value in original_env.items():
            if original_value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = original_value

import json
import glob

def train(args, learning_rates, batch_sizes, noise_stds, nmessage_passing_steps):
    """Runs the training process."""

    config_path = os.path.join(args.working_dir, 'config.json')
    with open(config_path, 'r') as f:
        original_config = json.load(f)

    for lr, batch_size, noise_std, nmessage_passing_steps in product(learning_rates, batch_sizes, noise_stds, nmessage_passing_steps):
        print(f"Starting training with learning rate: {lr}, batch size: {batch_size}, noise_std: {noise_std}, nmessage_passing_steps: {nmessage_passing_steps}")

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
        env["MACHINE_NAME"] = args.machine_name
        env["CUDA"] = "/usr/local/cuda-12/"
        env["PATH"] = env.get("PATH", "") + ":" + env["CUDA"]
        env["LD_LIBRARY_PATH"] = env.get("LD_LIBRARY_PATH", "") + ":" + env["CUDA"]
        env["OMP_NUM_THREADS"] = "1"
        
        model_suffix = f"{args.model_suffix}_lr{lr}_bs{batch_size}_ns{noise_std}_nmp{nmessage_passing_steps}_{args.machine_name}"
        model_path = os.path.join(args.working_dir, f'models.{model_suffix}', '')
        os.makedirs(model_path, exist_ok=True)

        # Check for existing model files to resume training
        existing_models = glob.glob(os.path.join(model_path, 'model-*.pt'))
        if existing_models:
            print(f"Found existing models in {model_path}. Resuming training.")
            args.model_file = "latest"
            args.train_state_file = "latest"
        else:
            print(f"No existing models found in {model_path}. Starting new training.")
            args.model_file = None # Ensure it's None if not resuming
            args.train_state_file = None # Ensure it's None if not resuming

        # Create a new config file with the updated learning rate
        new_config = original_config.copy()
        new_config['lr_init'] = lr
        new_config['noise_std'] = noise_std
        new_config['simulator_nmessage_passing_steps'] = nmessage_passing_steps
        config_file_path = os.path.join(model_path, 'config.json')
        with open(config_file_path, 'w') as f:
            json.dump(new_config, f, indent=4)
        # Ensure file is written before proceeding
        import time
        time.sleep(0.1)

        command_parts = [
            "python3 -m meshnet.train",
            f"--data_path={os.path.join(args.working_dir, 'dataset/')}",
            f"--model_path={model_path}",
            f"--output_path={os.path.join(args.working_dir, 'rollouts/')}",
            f"--batch_size={batch_size}",
            f"--nsave_steps={args.nsave_steps}",
            f"--ntraining_steps={args.ntraining_steps}"
        ]
        if args.model_file:
            command_parts.append(f"--model_file=\"{args.model_file}\"")
        if args.train_state_file:
            command_parts.append(f"--train_state_file=\"{args.train_state_file}\"")
        command = " ".join(command_parts)

        run_command(command, env)

def rollout(args):
    """Runs the rollout process."""
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

    for model_id in args.model_ids:
        model_file_name = f'model-{model_id}.pt'
        model_dir = os.path.join(args.working_dir, f'models.{args.model_suffix}', '')
        rollout_path = os.path.join(args.working_dir, f'rollouts.{args.model_suffix}/{model_file_name}')

        os.makedirs(rollout_path, exist_ok=True)
        
        log_file = os.path.join(rollout_path, 'rollout.log.txt')

        rollout_command = (
            f"python3 -m meshnet.train "
            f"--data_path={os.path.join(args.working_dir, 'dataset/')} "
            f"--model_path={model_dir} "
            f"--model_file={model_file_name} "
            f"--output_path={rollout_path} "
            f"--mode=rollout > {log_file}"
        )
        run_command(rollout_command, env)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run GNS training and rollout.")
    parser.add_argument("--working_dir", type=str, required=True, help="The working directory containing the dataset and config.json.")
    parser.add_argument("--mode", choices=['train', 'rollout', 'both'], help="Mode to run.")
    parser.add_argument("--model_suffix", type=str, help="Suffix for the model path.")
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU ID to use.")
    parser.add_argument("--machine_name", type=str, default="cotopaxi", help="Machine name (e.g., cotopaxi, knox).")
    parser.add_argument("--clear", action="store_true", help="Clear previous results.")
    
    # Training specific arguments
    parser.add_argument("--learning_rates", type=str, default="0.001", help="Comma-separated learning rates.")
    parser.add_argument("--batch_sizes", type=str, default="2", help="Comma-separated batch sizes.")
    parser.add_argument("--noise_stds", type=str, default="0.001", help="Comma-separated noise standard deviations.")
    parser.add_argument("--nmessage_passing_steps", type=str, default="10", help="Comma-separated number of message passing steps.")
    parser.add_argument("--ntraining_steps", type=int, default=10000000, help="Number of training steps.")
    parser.add_argument("--nsave_steps", type=int, default=100000, help="Number of steps between saving checkpoints.")

    # Rollout specific arguments
    parser.add_argument("--model_ids", type=str, default="1000000,2000000,3000000", help="Comma-separated model IDs for rollout.")
    parser.add_argument("--model_file", type=str, help="Model filename (.pt) to resume from. Can also use \"latest\" to default to newest file.")
    parser.add_argument("--train_state_file", type=str, help="Train state filename (.pt) to resume from. Can also use \"latest\" to default to newest file.")

    args = parser.parse_args()

    if args.clear:
        print("Clearing previous results...")
        import shutil
        models_path = os.path.join(args.working_dir, 'models.*')
        rollouts_path = os.path.join(args.working_dir, 'rollouts.*')
        for path in [models_path, rollouts_path]:
            for item in glob.glob(path):
                if os.path.isdir(item):
                    shutil.rmtree(item)
                elif os.path.isfile(item):
                    os.remove(item)
        if not args.mode:
            exit(0)

    if args.mode in ['train', 'rollout', 'both'] and not args.model_suffix:
        parser.error("argument --model_suffix is required when --mode is 'train', 'rollout', or 'both'.")

    learning_rates = [float(lr) for lr in args.learning_rates.split(',')]
    batch_sizes = [int(bs) for bs in args.batch_sizes.split(',')]
    noise_stds = [float(ns) for ns in args.noise_stds.split(',')]
    nmessage_passing_steps = [int(nmp) for nmp in args.nmessage_passing_steps.split(',')]
    args.model_ids = [int(mid) for mid in args.model_ids.split(',')]

    if args.mode in ['train', 'both']:
        train(args, learning_rates, batch_sizes, noise_stds, nmessage_passing_steps)
    
    if args.mode in ['rollout', 'both']:
        rollout(args)