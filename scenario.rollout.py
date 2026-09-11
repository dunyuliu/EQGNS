import os
import itertools
import argparse
import sys

# Add path for batch_rollout functionality
sys.path.append('meshnet')

# Define your lists of parameters

case = 5
use_batch_rollout = False  # Set to True to use batch rollout functionality

if case == 0:
    model_suffixes = {
        "/home/utig5/dliu/eq_rupture_gns/gns-sample/case4.200m.multi.stress.160scenarios.homo.a.Vw": [
            "nmp10.lr3e-5.b8.cotopaxi.r1"
            #"nmp10.lr3e-5.b8.n5e-3.cotopaxi.r1",
            #"nmp10.b4.cotopaxi.r1",
           # "nmp10.cotopaxi.r1"
        ]
    }
elif case == 1:
    model_suffixes = {
        "/home/utig5/dliu/eq_rupture_gns/work.test/case4.200m.multi.stress.homo.a.Vw": [
#            "r1_lr3e-05_bs2_ns0.005_nmp10_knox",
#            "r1_lr3e-05_bs2_ns0.005_nmp5_knox",
#            "r1_lr0.0001_bs2_ns0.005_nmp10_knox",
#            "r1_lr3e-05_bs2_ns0.02_nmp10_knox",
#            "r1_lr0.0001_bs2_ns0.005_nmp5_knox",   
#            "r1_lr3e-05_bs2_ns0.02_nmp5_knox",
            "r1_lr0.0001_bs2_ns0.02_nmp10_knox"  
#            "r1_lr0.0001_bs2_ns0.02_nmp5_knox"
        ]
        # Add more directories and their corresponding suffixes as needed
    }
elif case == 2:
    model_suffixes = {
        "/home/utig5/dliu/eq_rupture_gns/gns-sample/case4.200m.fractal.stress.homo.a.Vw": [
            "nmp10.cotopaxi.r1",
            "nmp5.cotopaxi.r1",
            "nmp15.cotopaxi.r1"
        ]
    }

elif case == 3:
    model_suffixes = {
        "/home/utig5/dliu/eq_rupture_gns/gns-sample/case4.200m.multi.stress.homo.a.Vw.case3.test": [
            "nmp10.cotopaxi.r1",
            "nmp5.cotopaxi.r1",
            "nmp15.cotopaxi.r1"
        ]
    }
elif case == 4:
    model_suffixes = {
        "/home/utig5/dliu/eq_rupture_gns/gns-sample/case4.200m.multi.stress.homo.a.Vw.case3.others.test": [
            "nmp10.cotopaxi.r1",
            "nmp5.cotopaxi.r1",
            "nmp15.cotopaxi.r1"
        ]
    }
elif case == 5:
    model_suffixes = {
        "/home/utig5/dliu/eq_rupture_gns/gns-sample/case4.200m.multi.stress.homo.a.Vw": [
             "nmp10.cotopaxi"
#            "nmp10.cotopaxi.r1"
#            "nmp5.cotopaxi.r1",
#            "nmp15.cotopaxi.r1"
        ]
    }
elif case == 6:
    model_suffixes = {
        "/home/utig5/dliu/dynamo_gns/inbox/from_gns_clone/work.dynamo/D1.Ra.normalizedRa/": [
        "r0_lr0.0001_bs2_ns0.02_nmp10_cotopaxi",
        "r0_lr0.0001_bs4_ns0.02_nmp10_cotopaxi",
        "r0_lr3e-05_bs4_ns0.02_nmp10_cotopaxi",
        "r0_lr3e-05_bs2_ns0.02_nmp10_cotopaxi"
        ]
    }
elif case == 7:
    model_suffixes = {
        "/home/staff/dliu/eq_rupture_gns/gns-sample/case3.200m.homo.a.Vw.others/": [
        "nmp10.cotopaxi"
        ]
    }
elif case == 8:
    model_suffixes = {
        "/home/staff/dliu/eq_rupture_gns/gns-sample/case3.200m.homo.a.Vw.others/": [
        "nmp10.cotopaxi"
        ]
    }




model_id = "3000000"  # Set your model ID here
gpu_id = 0 # Set GPU id to use

# Batch rollout parameters (when use_batch_rollout = True)
batch_size = 4  # Number of .pkl files to process in each batch
pkl_path = None  # Path to .pkl files for batch processing
data_path = "/home/utig5/dliu/dynamo_gns/inbox/from_gns_clone/work.dynamo/D1.Ra.normalizedRa/dataset"  # Path to .npz files for batch processing
output_path = "rollouts/"  # Output path for rollouts

if use_batch_rollout:
    # Use batch rollout functionality
    try:
        from batch_rollout import main as batch_main
        from absl import flags
        from absl import app
        import sys
        
        print("Using batch rollout functionality...")
        
        # Collect all models for batch processing
        all_models = []
        for working_dir, suffixes in model_suffixes.items():
            for suffix in suffixes:
                all_models.append((working_dir, suffix))

        print(f"\nProcessing {len(all_models)} models in batch: {[f'{wd}:{suf}' for wd, suf in all_models]}")

        # Create single batch run with all models
        original_argv = sys.argv.copy()

        # Use first working_dir as base and pass all model suffixes
        first_working_dir = all_models[0][0]
        all_suffixes = [suffix for _, suffix in all_models]

        batch_argv = [
            'batch_rollout.py',
            '--mode', 'rollout',
            '--working_dir', first_working_dir,
            '--model_suffix', ','.join(all_suffixes),  # Pass comma-separated list
            '--model_ids', str(model_id),
            '--gpu_id', str(gpu_id),
            '--batch_size', str(batch_size)
        ]

        # Add data source if specified
        if pkl_path:
            batch_argv.extend(['--pkl_path', pkl_path])
        if data_path:
            batch_argv.extend(['--data_path', data_path])

        # Set sys.argv and run
        sys.argv = batch_argv

        # Run batch rollout using absl.app.run
        try:
            app.run(batch_main)
        except SystemExit:
            pass  # Ignore SystemExit from absl

        # Restore original sys.argv
        sys.argv = original_argv
                
    except ImportError as e:
        print(f"Error importing batch_rollout: {e}")
        print("Falling back to standard rollout...")
        use_batch_rollout = False

if not use_batch_rollout:
    # Use standard rollout functionality (original behavior)
    print("Using standard rollout functionality...")
    
    # Loop over directories and their suffixes
    for working_dir, suffixes in model_suffixes.items():
        for suffix in suffixes:
            command = (f"python /home/utig5/dliu/eq_rupture_gns/run.process.gns.py "
                  f"--working_dir {working_dir} "
                  f"--mode rollout "
                  f"--model_suffix {suffix} "
                  f"--model_ids {model_id} "
                  f"--gpu_id {gpu_id} ")
                  #f"--cpu_only ")
            print(command)
            # Execute the command
            os.system(command)
