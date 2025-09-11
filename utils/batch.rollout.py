import os
import itertools

# Define your lists of parameters

case = 1

if case == 0:
    model_suffixes = {
        "/home/utig5/dliu/gns/gns/gns-sample/case4.200m.multi.stress.160scenarios.homo.a.Vw": [
            "nmp10.lr3e-5.b8.cotopaxi.r1",
            "nmp10.lr3e-5.b8.n5e-3.cotopaxi.r1",
            "models.nmp10.lr3e-5.b12.cotopaxi.r1",
            "nmp10.b4.cotopaxi.r1",
            "nmp10.cotopaxi.r1"
        ]
    }
elif case == 1:
    model_suffixes = {
        "/home/utig5/dliu/gns/gns/work.test/work.test/case4.200m.multi.stress.homo.a.Vw": [
            "r1_lr3e-05_bs2_ns0.005_nmp10_knox",
            "r1_lr3e-05_bs2_ns0.005_nmp5_knox",
            "r1_lr0.0001_bs2_ns0.005_nmp10_knox",
            "r1_lr3e-05_bs2_ns0.02_nmp10_knox",
            "r1_lr0.0001_bs2_ns0.005_nmp5_knox",   
            "r1_lr3e-05_bs2_ns0.02_nmp5_knox",
            "r1_lr0.0001_bs2_ns0.02_nmp10_knox",  
            "r1_lr0.0001_bs2_ns0.02_nmp5_knox"
        ]
        # Add more directories and their corresponding suffixes as needed
    }



model_id = "1000000"  # Set your model ID here
gpu_id = 1 # Set GPU id to use

# Loop over directories and their suffixes
for working_dir, suffixes in model_suffixes.items():
    for suffix in suffixes:
        command = (f"python /home/utig5/dliu/gns/gns/run.process.gns.py "
              f"--working_dir {working_dir} "
              f"--mode rollout "
              f"--model_suffix {suffix} "
              f"--model_ids {model_id} "
              f"--gpu_id {gpu_id} ")
        print(command)
        # Execute the command
        os.system(command)
