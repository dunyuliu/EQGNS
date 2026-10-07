import os
import json
import shutil
import subprocess
import itertools
import csv
from pathlib import Path
from datetime import datetime

LOG_FILE = "./training_runs_log.csv"

# ------------------- Functions ------------------- #

def prepare_run_folder(data, model_suffix, learning_rate, num_message_passing_steps, noise_level):
    """
    Prepare folder and config.json for training
    """
    SCRATCH = "./gns-sample"
    data_path = Path(SCRATCH) / data / "dataset"
    run_folder = Path(SCRATCH) / data
    model_path = Path(SCRATCH) / data / f"models.{model_suffix}"
    rollout_path = Path(SCRATCH) / data / f"rollouts.{model_suffix}"
    model_path.mkdir(parents=True, exist_ok=True)
    rollout_path.mkdir(parents=True, exist_ok=True)

    # Copy config.json
    base_config = run_folder / "config.json"
    run_config = model_path / "config.json"
    shutil.copyfile(base_config, run_config)
    # Update parameters
    with open(run_config, 'r') as f:
        config = json.load(f)
    #config['batch_size'] = batch_size # through command line
    config['lr_init'] = learning_rate
    config['noise_std'] = noise_level
    config['simulator_nmessage_passing_steps'] = num_message_passing_steps
    #config['ntraining_steps'] = 1_000_000
    with open(run_config, 'w') as f:
        json.dump(config, f, indent=4)

    return run_folder, run_config, data_path, model_path, output_path

def run_training(run_config, data_path, model_path, output_path, gpu_id, batch_size, total_training_steps):
    """
    Run training with the prepared config
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ["OMP_NUM_THREADS"] = "1"
    train_cmd = [
        "python3", "-m", "meshnet.train",
        f"--data_path={data_path}",
        f"--model_path={model_path}",
        f"--output_path={output_path}",
        f"--nsave_steps=100000",
        f"--batch_size={batch_size}",
        f"--ntraining_steps={int(total_training_steps)}",
    ]
    print(f"\n=== Training Run: {run_config.parent} ===\n")
    subprocess.run(train_cmd, check=True)

def rollout_model(data, model_suffix, gpu_id, model_file, testset_name, render_or_not=False):
    """
    Run rollout for a trained model, dynamically rendering based on existing .pkl files
    """
    TMP_DIR = Path.cwd() / "gns-sample"
    DATA_PATH = TMP_DIR / data / "dataset"
    MODEL_PATH = TMP_DIR / data / f"models.{model_suffix}"
    ROLLOUT_PATH = TMP_DIR / data / f"rollouts.{model_suffix}/{model_file}"

    # Create rollout folder
    if ROLLOUT_PATH.exists():
        shutil.rmtree(ROLLOUT_PATH)
    ROLLOUT_PATH.mkdir(parents=True, exist_ok=True)

    # Copy test dataset and metadata
    #shutil.copyfile(Path("dataset_archive") / testset_name, DATA_PATH / "test.npz")
    shutil.copyfile(DATA_PATH / "testset_metadata.json", ROLLOUT_PATH / "testset_metadata.json")

    # Skip if .pkl files exist (already rolled out)
    existing_pkls = list(ROLLOUT_PATH.glob("*.pkl"))
    if existing_pkls:
        print(f".pkl files exist for {model_file}, skipping rollout.")
        return

    # Run rollout
    train_cmd = [
        "python3", "-m", "meshnet.train",
        f"--data_path={DATA_PATH}",
        f"--model_path={MODEL_PATH}",
        f"--model_file={model_file}",
        f"--output_path={ROLLOUT_PATH}",
        "--mode=rollout"
    ]
    log_file = ROLLOUT_PATH / "rollout.log.txt"
    with open(log_file, "a") as f:
        subprocess.run(train_cmd, check=True, stdout=f, stderr=subprocess.STDOUT)

    if render_or_not == True:
        # Dynamically get the number of .pkl files and render
        rollout_pkls = sorted(ROLLOUT_PATH.glob("*.pkl"))
        print(f"Found {len(rollout_pkls)} rollout .pkl files, rendering...")
        for pkl_file in rollout_pkls:
            rollout_name = pkl_file.stem  # filename without extension
        render_cmd = [
            "python3", "-m", "meshnet.render",
            f"--rollout_dir={ROLLOUT_PATH}",
            f"--rollout_name={rollout_name}"
        ]
        with open(log_file, "a") as f:
            subprocess.run(render_cmd, check=True, stdout=f, stderr=subprocess.STDOUT)

def log_run(log_file, data, batch_size, learning_rate, num_message_passing_steps, noise_level, run_folder, status):
    """Append run info to CSV log"""
    file_exists = os.path.isfile(log_file)
    with open(log_file, mode='a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if not file_exists:
            writer.writerow(["timestamp", "data", "batch_size", "learning_rate", "num_message_passing_steps", "noise_level", "run_folder", "status"])
        writer.writerow([
            datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            data, batch_size, learning_rate, noise_level, str(run_folder), status
        ])

# ------------------- Main Pipeline ------------------- #

if __name__ == "__main__":
    data = "case4.200m.multi.stress.homo.a.Vw.batchrun"
    total_training_steps = 1e5
    gpu_id = 3
    render_or_not = False
    testset_name = "your_testset_name_here"  # specify your test dataset

    message_passing_steps = [5, 8, 10]
    batch_sizes = [4, 8, 12]
    learning_rates = [3e-5, 5e-5, 1e-4]
    noise_levels = [2e-2, 5e-3]

    for bs, lr, nmp, noise in itertools.product(batch_sizes, learning_rates, message_passing_steps, noise_levels):
        model_suffix = f"nmp{nmp}.b{bs}.lr{lr}.n{noise}.cotopaxi.r1"
        try:
            run_folder, run_config, data_path, model_path, output_path = prepare_run_folder(
                data, model_suffix, lr, nmp, noise
            )
            run_training(run_config, data_path, model_path, output_path, gpu_id, bs, total_training_steps)
            log_run(LOG_FILE, data, bs, lr, noise, run_folder, status="success")

            # Run rollout for the trained model
            # Automatically pick latest model in model_path
            model_files = sorted(model_path.glob("*.pt"))
            if not model_files:
                print(f"No model found in {model_path}, skipping rollout.")
                continue
            latest_model_file = model_files[-1].name  # pick the last (latest)
            rollout_model(data, model_suffix, gpu_id, latest_model_file, testset_name, render_or_not=render_or_not)

        except subprocess.CalledProcessError as e:
            print(f"❌ Training failed for bs={bs}, lr={lr}, noise={noise}")
            log_run(LOG_FILE, data, bs, lr, nmp, noise, run_folder, status="failed")