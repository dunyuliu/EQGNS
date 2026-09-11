
import argparse
import os
import subprocess

def main():
    """
    This script automates the training of a model using meshnet.
    It sets up the necessary environment and executes the training command
    with specified parameters.
    """
    parser = argparse.ArgumentParser(
        description="""
        Train a model using meshnet.
        This script is a Python replacement for train.sh.
        Example Usage:
        python train_cli.py case3.200m nmp10.cotopaxi 0 2
        """,
        formatter_class=argparse.RawTextHelpFormatter
    )

    parser.add_argument(
        'data',
        type=str,
        help="""Name of the dataset to use.
Available datasets include:
- case3.200m
- case4.200m.multi.stress
"""
    )
    parser.add_argument(
        'model_suffix',
        type=str,
        help="Suffix for the model path."
    )
    parser.add_argument(
        'gpu_id',
        type=str,
        help="The ID of the GPU to use for training."
    )
    parser.add_argument(
        'batch_size',
        type=int,
        help="Batch size for training."
    )
    parser.add_argument(
        '--nsave_steps',
        type=int,
        default=100000,
        help="Number of steps between saving checkpoints."
    )
    parser.add_argument(
        '--ntraining_steps',
        type=int,
        default=10000000,
        help="Total number of training steps."
    )
    parser.add_argument(
        '--model_file',
        type=str,
        default=None,
        help="Load a specific model file (e.g., 'latest')."
    )
    parser.add_argument(
        '--train_state_file',
        type=str,
        default=None,
        help="Load a specific training state file (e.g., 'latest')."
    )

    args = parser.parse_args()

    scratch = "./gns-sample"
    cuda_path = "/usr/local/cuda-12/"

    env = os.environ.copy()
    env['PATH'] = f"{env.get('PATH', '')}:{cuda_path}"
    env['LD_LIBRARY_PATH'] = f"{env.get('LD_LIBRARY_PATH', '')}:{cuda_path}"
    env['OMP_NUM_THREADS'] = '1'
    env['CUDA_VISIBLE_DEVICES'] = args.gpu_id

    cmd = [
        'python3', '-m', 'meshnet.train',
        '--data_path', f'{scratch}/{args.data}/dataset/',
        '--model_path', f'{scratch}/{args.data}/models.{args.model_suffix}/',
        '--output_path', f'{scratch}/{args.data}/rollouts/',
        '--batch_size', str(args.batch_size),
        '--nsave_steps', str(args.nsave_steps),
        '--ntraining_steps', str(args.ntraining_steps),
    ]

    if args.model_file:
        cmd.extend(['--model_file', args.model_file])
    if args.train_state_file:
        cmd.extend(['--train_state_file', args.train_state_file])

    print("Executing command:")
    print(' '.join(cmd))

    try:
        subprocess.run(cmd, check=True, env=env)
    except subprocess.CalledProcessError as e:
        print(f"Error executing command: {e}")
        print(f"Stderr: {e.stderr}")
        print(f"Stdout: {e.stdout}")


if __name__ == '__main__':
    main()
