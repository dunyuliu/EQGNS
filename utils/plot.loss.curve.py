#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt

font = {'family': 'serif',
        'weight': 'bold',
        'size': 12}

plt.rc('font', **font)
plt.rcParams['axes.labelweight'] = font['weight']     # Ensures bold axis labels

step = 600000
nstep = 5
#model_path = "meshnet.eq/all.nodes.mark0/models/"
#model_path = "case4.np.lr.1e-6.eq.meshnet.results/"
#model_path = "case4.40.np.lr.1e-4.eq.meshnet.results/"
#model_path = "case4.40.np.lr.1e-4.nmp.9/"
#model_path = "case4.two.stress/nmp5/"
#model_path = "dynamo/nmp10/"
#model_path = "../results/case4.200m.multi.stress/nmp10.knox/"

model_list = ["../results/case3.200m.homo.a.Vw/nmp10.cotopaxi/", 
              "../results/case4.200m.multi.stress.homo.a.Vw/nmp10.cotopaxi/"]

for model_path in model_list:
    loss_log = np.loadtxt(model_path+'loss_log.txt')
    #valid_loss_log = np.loadtxt(model_path+'valid_loss_log.txt')
    loss_log_coarse = []
    itag = 0
    for i in range(loss_log.shape[0]):
        if i%nstep == 0:
            loss_log_coarse.append(loss_log[i,:])
            itag += 1

    loss_log_coarse = np.array(loss_log_coarse)

    fig, ax = plt.subplots(figsize=(5,4))
    ax.plot(loss_log_coarse[:,0], loss_log_coarse[:,1], label='train')
    ax.plot(loss_log_coarse[:,0], loss_log_coarse[:,2], label='valid')

    ax.set_yscale('log')
    ax.set_xlabel("Training steps")
    ax.set_ylabel(f"Loss evaluated every {1000*nstep} steps")
    fig.tight_layout()
    ax.legend()

    fig.savefig(f"{model_path}train_valid_loss_per_{1000*nstep}_steps.png", dpi=300)