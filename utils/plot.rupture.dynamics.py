#! /usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
import pickle
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import matplotlib.patches as patches
from mpl_toolkits.axes_grid1 import ImageGrid

font = {'family': 'serif',
        'weight': 'bold',
        'size': 10}
plt.rc('font', **font)
plt.rcParams['axes.titlesize'] = 10
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 10
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['axes.titlepad'] = 5
plt.rcParams['axes.labelpad'] = 5
plt.rcParams['xtick.major.pad'] = 5
plt.rcParams['ytick.major.pad'] = 5
plt.rcParams['xtick.minor.pad'] = 5
plt.rcParams['ytick.minor.pad'] = 5
plt.rcParams['axes.linewidth'] = 0.5
plt.rcParams['lines.linewidth'] = 0.5
plt.rcParams['lines.markersize'] = 2
plt.rcParams['figure.figsize'] = (6, 4)
plt.rcParams['figure.dpi'] = 600
# Define discrete stress levels and matching colors
stress_bins = [0, 0.25, 0.5, 0.75, 1.0]
colors = plt.cm.viridis([0.0, 0.25, 0.5, 0.75, 1.0])

# Build a mapping function
def stress_to_color(s):
    if s ==0:
        return colors[0]
    elif s == 0.25:
        return colors[1]
    elif s == 0.5:
        return colors[2]
    elif s == 0.75:
        return colors[3]
    elif s == 1.0:
        return colors[4]

def stress_to_color_mpa(s):
    if s =='35MPa':
        return colors[0]
    elif s == '40MPa':
        return colors[1]
    elif s == '45MPa':
        return colors[2]
    elif s == '50MPa':
        return colors[3]
    elif s == '55MPa':
        return colors[4]
      
case = 0
if case ==0: 
    case_path = "../results/case3.200m/nmp10.knox/model-3000000.pt/"
    test_metadata = os.path.join("./results/case3.200m/dataset/case3.200m.test.metadata.json")
elif case == 1:
    case_path = "../results/case4.200m.multi.stress/nmp10.knox/model-2000000.pt/"
    test_metadata = os.path.join("../results/case4.200m.multi.stress/dataset/case4.200m.multi.stress.test.metadata.json")
elif case == 2:
    case_path = "../results/case4.200m.multi.asp/nmp10.cotopaxi/model-4000000.pt/"
    test_metadata = os.path.join("./case4.200m.multi.asp/dataset/case4.200m.checkerboard.stress.test.metadata.json")
elif case == 3:
    case_path = "./case3.200m.others/nmp10.cotopaxi/model-2000000.pt/"
    test_metadata = "./case3.200m.others/dataset/case3.200m.others.test.metadata.json"
    #test_metadata = os.path.join("./case3.200m.others/dataset/case4.200m.checkerboard.stress.test.metadata.json")
case_id = int(sys.argv[1])
case_name = case_path + f"rollout_{sys.argv[1]}.pkl"

is_asperity = False
if os.path.exists(test_metadata):
    with open(test_metadata) as f:
        test_metadata = json.load(f)
    asp_rect = []
    if 'asperity_location_km' in test_metadata[0].keys():
        is_asperity = True
        item = test_metadata[case_id]
        # Extract asperity info
        asp_x, asp_y, half_size, stress1, stress2 = item["asperity_location_km"]
        full_size = 2 * half_size
        color = stress_to_color(stress2)
        asp_rect.append(patches.Rectangle(
            (asp_x - half_size, asp_y - half_size),
            full_size, full_size,
            linewidth=3,
            edgecolor=color,
            facecolor='none'
        ))
    if 'asperities' in test_metadata[0].keys():
        is_asperity = True
        item = test_metadata[case_id]
        # Extract asperity info
        asp_rect = []
        for asp in item["asperities"]:
            asp_x, asp_y = asp['asperity_location_km']
            stress = asp['stress_level']
            half_size = asp['asperity_half_square_size_km']
            full_size = 2 * half_size
            color = stress_to_color_mpa(stress)
            asp_rect.append(patches.Rectangle(
                (asp_x - half_size, asp_y - half_size),
                full_size, full_size,
                linewidth=3,
                edgecolor=color,
                facecolor='none'
            ))

print(f"processing {case_name}")  # Removed as "processing" is not defined
dt = 0.0167777 # 0.0167777 seconds per timestep
sliprate_threshold = 5 # slip rate threshold for rupture time
vmin, vmax = 0, 10

# read rollout data
with open(case_name, 'rb') as f:
    result = pickle.load(f)
ground_truth_vel = np.concatenate((result["initial_velocities"], result["ground_truth_rollout"]))
predicted_vel = np.concatenate((result["initial_velocities"], result["predicted_rollout"]))

# compute velocity magnitude
ground_truth_vel_magnitude = np.linalg.norm(ground_truth_vel, axis=-1)
predicted_vel_magnitude = np.linalg.norm(predicted_vel, axis=-1)
velocity_result = {
    "ground_truth": ground_truth_vel_magnitude,
    "prediction": predicted_vel_magnitude
}

# variables for render
n_timesteps = len(ground_truth_vel_magnitude)
triang = tri.Triangulation(result["node_coords"][0][:, 0]/1e3, result["node_coords"][0][:, 1]/1e3)
nnode = len(result["node_coords"][0])

def extract_timeseries(velocity_result, triang, node_loc):
    """
    Extract time series of velocity for a given node location.
    """
    # Find the index of the node closest to the specified location

    points = triang.x, triang.y
    nodes = np.vstack(points).T

    # Find the nearest node index
    distances = np.linalg.norm(nodes - node_loc, axis=1)
    node_id = np.argmin(distances)

    if node_id is None:
        raise ValueError("Node location not found in triangulation.")
    
    # Extract the time series for the specified node
    timeseries = {sim: vel[:, node_id] for sim, vel in velocity_result.items()}
    
    return timeseries

def plot_slip_rate_snapshots(velocity_result, triang, timestep_id, case_id, case_path, mode='ground_truth', vmin=0, vmax=10):
    """
    Plot slip rate snapshots for each simulation at each time step.
    mode: 'ground_truth' or 'prediction' or 'both'
    """
    # Create a figure with subplots
    

    # Loop through each simulation and plot the velocity
    if mode == 'both':
        fig = plt.figure(figsize=(6, 6))
        grid = ImageGrid(fig, 111,
                            nrows_ncols=(2, 1),
                            axes_pad=0.15,
                            share_all=True,
                            cbar_location="right",
                            cbar_mode="single",
                            cbar_size="1.5%",
                            cbar_pad=0.15)
        for j, (sim, vel) in enumerate(velocity_result.items()):
            #grid[j].triplot(triang, 'o-', color='k', ms=0.5, lw=0.3)
            handle = grid[j].tripcolor(triang, vel[timestep_id], vmax=vmax, vmin=vmin)
            fig.colorbar(handle, cax=grid.cbar_axes[0])
            grid[j].set_title(sim)
    else:
        fig = plt.figure(figsize=(6, 3))
        grid = ImageGrid(fig, 111, nrows_ncols=(1, 1))
        handle = grid[0].tripcolor(triang, velocity_result[mode][timestep_id], vmax=vmax, vmin=vmin)
        grid[0].set_xticks([])
        grid[0].set_yticks([])
        #fig.colorbar(handle, cax=grid.cbar_axes[0])

    fig.savefig(os.path.join(case_path, f"a.Sliprate_snapshot_time_{mode}_{timestep_id}_rollout_{case_id}.png"), dpi=600)

def get_rupture_time(sliprate_hist, dt, threshold=0.001, unreachable_val=1000):
    """
    Compute rupture time for each node based on when the slip rate exceeds a threshold.

    Parameters:
        sliprate_hist: np.ndarray of shape (n_timesteps, n_nodes)
        dt: timestep duration
        threshold: slip rate threshold to define rupture
        unreachable_val: value to assign if node never ruptures

    Returns:
        rupture_time: np.ndarray of shape (n_nodes,) with rupture time per node
    """
    n_timesteps, n_nodes = sliprate_hist.shape
    rupture_time = np.full(n_nodes, unreachable_val, dtype=float)

    for it in range(n_timesteps):
        active = (rupture_time == unreachable_val) & (sliprate_hist[it] > threshold)
        #print(f"Timestep {it}: {np.sum(active)} nodes ruptured")
        rupture_time[active] = it * dt + 1.2

    return rupture_time

rpt_ground_truth = get_rupture_time(velocity_result["ground_truth"], dt, sliprate_threshold, 1000)
rpt_predicted = get_rupture_time(velocity_result["prediction"], dt, sliprate_threshold, 1000)

def plot_combined_rpt_contours(triang, rpt_gt, rpt_pred, filename, asp_rect=None):
    """
    Usage: plot_combined_rpt_contours(triang, rpt_gt, rpt_pred, filename, asp_rect=None)
    Plot combined rupture time contours for ground truth and prediction.
    Parameters:
        triang: triangulation object for the mesh
        rpt_gt: rupture time for ground truth
        rpt_pred: rupture time for prediction
        filename: path to save the plot
        asp_rect: list of patches for asperities (optional)
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    # Mask unreachable points
    mask_gt = np.ma.masked_where(rpt_gt >= 999, rpt_gt)
    mask_pred = np.ma.masked_where(rpt_pred >= 999, rpt_pred)

    levels = np.arange(0, 15, 0.5)
    # Ground truth: solid black lines
    mask_gt = mask_gt.filled(0)
    cs_gt = ax.tricontour(triang, mask_gt, levels=levels, colors='black', linewidths=1.0)
    contour_labels = ax.clabel(cs_gt, fmt=lambda x: f'GT {x:.1f}', fontsize=10, inline=True, inline_spacing=20)
    for label in contour_labels:
        label.set_position(label.get_position() + np.array([0, 0.2]))  # Adjust the offset as needed

    # Prediction: dashed red lines
    mask_pred = mask_pred.filled(0)
    cs_pred = ax.tricontour(triang, mask_pred, levels=levels, colors='red', linestyles='--', linewidths=1.0)
    contour_labels = ax.clabel(cs_pred, fmt=lambda x: f'PR {x:.1f}', fontsize=10, inline=True, inline_spacing=10)
    for label in contour_labels:
        label.set_position(label.get_position() + np.array([0, -0.2]))  # Adjust the offset as needed

    if asp_rect is not None:
        for rect in asp_rect:
            ax.add_patch(rect)
        legend_patches = [
                patches.Patch(color=colors[i], label=f'{stress_bins[i]:.2f}')
                for i in range(len(stress_bins))
            ]
        ax.legend(handles=legend_patches, title="Normalized Stress", fontsize=9, title_fontsize=10, loc='best')
    #ax.set_title("Rupture Time: Ground Truth vs Prediction")
    ax.grid(True)
    ax.set_xlabel("Strike (m)")
    ax.set_ylabel("Dip (m)")
    ax.set_aspect('equal', adjustable='box')
    fig.tight_layout()
    fig.savefig(filename, dpi=600)
    plt.close(fig)

plot_combined_rpt_contours(
    triang,
    rpt_ground_truth,
    rpt_predicted,
    os.path.join(case_path, f"a.rupture_time_comparison_rollout_{case_id}.png"),
    asp_rect=asp_rect if is_asperity else None
)
for mode in ['ground_truth', 'both']:
    plot_slip_rate_snapshots(velocity_result, triang, int(0/dt), case_id, case_path, mode=mode)
    plot_slip_rate_snapshots(velocity_result, triang, int(1/dt), case_id, case_path, mode=mode)
    plot_slip_rate_snapshots(velocity_result, triang, int(2/dt), case_id, case_path, mode=mode)
    plot_slip_rate_snapshots(velocity_result, triang, int(3/dt), case_id, case_path, mode=mode)
    plot_slip_rate_snapshots(velocity_result, triang, int(4/dt), case_id, case_path, mode=mode)
    plot_slip_rate_snapshots(velocity_result, triang, int(5/dt), case_id, case_path, mode=mode)
   
print(np.max((rpt_predicted)))
print("Ground truth slip max:", np.max(velocity_result["ground_truth"]))
print("Prediction slip max:", np.max(velocity_result["prediction"]))

# Plotting the slip rate time series for a specific node location

def plot_slip_rate_time_series(velocity_result, triang, node_loc, dt, case_id, case_path):
    """
    usage: plot_slip_rate_time_series(velocity_result, triang, node_loc, dt, case_id, case_path)
    Extract and plot the slip rate time series for a specific node location.
    Parameters:
        velocity_result: dict containing velocity data for each simulation
        triang: triangulation object for the mesh
        node_loc: tuple (x, y) specifying the node location
        dt: time step duration
        case_id: case ID for saving the plot
        case_path: path to save the plot
    """
    timeseries = extract_timeseries(velocity_result, triang, node_loc)
    fig, ax = plt.subplots(figsize=(5, 5))
    for sim, vel in timeseries.items():
        ax.plot(np.arange(len(vel)) * dt, vel, label=sim, linewidth=2)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Slip Rate (m/s)")
    ax.set_title(f"Slip Rate Time Series at Node Location {node_loc}")
    ax.legend()
    plt.savefig(os.path.join(case_path, f"b.Sliprate_time_series_node_{node_loc[0]}_{node_loc[1]}_rollout_{case_id}.png"), dpi=600)
    plt.close(fig)

plot_slip_rate_time_series(velocity_result, triang, (-1, -7), dt, case_id, case_path)
plot_slip_rate_time_series(velocity_result, triang, (3, -3), dt, case_id, case_path)
plot_slip_rate_time_series(velocity_result, triang, (7, -7), dt, case_id, case_path)
plot_slip_rate_time_series(velocity_result, triang, (-1, -3), dt, case_id, case_path)
plot_slip_rate_time_series(velocity_result, triang, (3, -7), dt, case_id, case_path)
plot_slip_rate_time_series(velocity_result, triang, (7, -3), dt, case_id, case_path)