from prepare_eqdyna_4gns import create_train_data
import numpy as np
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../..")

import json
# .npz names for 100m resolution
dataset_root = "dataset.case3.200m.others/"
metadata_setname = "case3.200m.other"
set1 = {'model_name':'tpv104.200m.H14.large',
            'hypocenter_location_km': [3,-4]}
ntag = 0
set_dict = {}
selected_cases = []
selected_cases.append(set1)
for case in selected_cases:
    model_name = case['model_name']
    model_path = dataset_root+model_name
    print('Processing model', model_path)
    particle, meshnet, _, _, _, _ = create_train_data(model_path)
    traj_name = "trajectory"+str(ntag)
    set_dict[traj_name] = meshnet
    ntag+=1

np.savez(dataset_root+metadata_setname,**set_dict)

with open(dataset_root+metadata_setname+".metadata.json", "w") as f:
    json.dump(selected_cases, f, indent=4)


