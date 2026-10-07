#! /usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced rupture dynamics visualization with comprehensive error analysis.
Supports processing single members or all members (0-30) with dedicated output folders.

Usage:
    python plot.rupture.dynamics_mod.py <member_id>        # Process single member
    python plot.rupture.dynamics_mod.py --all              # Process all members 0-30
    python plot.rupture.dynamics_mod.py --help             # Show help
"""

import os
import sys
import pickle
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import matplotlib.patches as patches
from mpl_toolkits.axes_grid1 import ImageGrid
from scipy import stats
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Global configuration
font = {'family': 'serif', 'weight': 'bold', 'size': 12}
plt.rc('font', **font)
plt.rcParams['axes.labelweight'] = font['weight']
plt.rcParams['axes.labelsize'] = font['size']
plt.rcParams['axes.titleweight'] = font['weight']

stress_bins = [0, 0.25, 0.5, 0.75, 1.0]
colors = plt.cm.viridis([0.0, 0.25, 0.5, 0.75, 1.0])

# Analysis parameters
DT = 0.0167777
SLIPRATE_THRESHOLD = 0.1
VMIN, VMAX = 0, 10

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
      
def get_case_config(case=4):
    """Get case configuration including paths and metadata."""
    configs = {
        0: {
            "case_path": "../results/case3.200m.homo.a.Vw/nmp10.cotopaxi/model-3000000.pt/",
            "test_metadata": ""
        },
        1: {
            "case_path": "../results/case3.200m.homo.a.Vw.others/nmp10.cotopaxi/model-3000000.pt/",
            "test_metadata": ""
        },
        2: {
            "case_path": "../results/case4.200m.multi.stress.homo.a.Vw/nmp10.cotopaxi/model-3000000.pt/",
            "test_metadata": "../results/case4.200m.multi.stress.homo.a.Vw/dataset/case4.200m.multi.stress.test.metadata.json"
        },
        3: {
            "case_path": "../results/case4.200m.multi.asp.homo.a.Vw/nmp10.cotopaxi/model-3000000.pt/",
            "test_metadata": "../results/case4.200m.multi.asp.homo.a.Vw/dataset/case4.200m.checkerboard.stress.test.metadata.json"
        },
        4: {
            "case_path": "../results/case4.200m.fractal.stress.homo.a.Vw/nmp10.cotopaxi/model-3000000.pt/",
            "test_metadata": ""
        },
        5: {
            "case_path": "../results/case4.200m.multi.stress.160scenarios.homo.a.Vw/nmp10.cotopaxi.r1/model-5000000.pt/",
            "test_metadata": ""
        },
        6: {
            "case_path": "../results/case4.200m.multi.stress.160scenarios.homo.a.Vw/nmp10.b4.cotopaxi.r1/model-3000000.pt/",
            "test_metadata": ""
        },
        7: {
            "case_path": "../results/case4.200m.multi.stress.160scenarios.homo.a.Vw/nmp10.lr3e-5.b8.cotopaxi.r1/model-3000000.pt/",
            "test_metadata": ""
        },
        8: {
            "case_path": "../results/case4.200m.multi.stress.homo.a.Vw.case3.test/nmp10.cotopaxi.r1/model-3000000.pt/",
            "test_metadata": ""
        },
        9: {
            "case_path": "../results/case4.200m.multi.stress.homo.a.Vw.case3.others.test/nmp5.cotopaxi.r1/model-3000000.pt/",
            "test_metadata": ""
        },
        10: {
            "case_path": "../results/case4.200m.multi.stress.160scenarios.homo.a.Vw.case3.test/nmp10.cotopaxi.r1/model-3000000.pt/",
            "test_metadata": ""
        }
    }
    return configs.get(case, configs[4])

def detect_available_members(case_path):
    """Detect available pkl files and return the list of member IDs."""
    if not os.path.exists(case_path):
        return []
    
    member_ids = []
    for filename in os.listdir(case_path):
        if filename.startswith("rollout_") and filename.endswith(".pkl"):
            try:
                # Extract member ID from filename like "rollout_0.pkl"
                member_id = int(filename.replace("rollout_", "").replace(".pkl", ""))
                member_ids.append(member_id)
            except ValueError:
                # Skip files that don't follow the expected pattern
                continue
    
    return sorted(member_ids)

def load_asperity_info(test_metadata_path, case_id):
    """Load asperity information from metadata file."""
    is_asperity = False
    asp_rect = []
    
    if not os.path.exists(test_metadata_path):
        return is_asperity, asp_rect
    
    with open(test_metadata_path) as f:
        test_metadata = json.load(f)
    
    if 'asperity_location_km' in test_metadata[0].keys():
        is_asperity = True
        item = test_metadata[case_id]
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
    
    return is_asperity, asp_rect

def load_rollout_data(case_path, case_id):
    """Load rollout data for a specific case."""
    case_name = os.path.join(case_path, f"rollout_{case_id}.pkl")
    
    if not os.path.exists(case_name):
        raise FileNotFoundError(f"Rollout file not found: {case_name}")
    
    print(f"Processing {case_name}")
    
    with open(case_name, 'rb') as f:
        result = pickle.load(f)
    
    ground_truth_vel = np.concatenate((result["initial_velocities"], result["ground_truth_rollout"]))
    predicted_vel = np.concatenate((result["initial_velocities"], result["predicted_rollout"]))
    
    ground_truth_vel_magnitude = np.linalg.norm(ground_truth_vel, axis=-1)
    predicted_vel_magnitude = np.linalg.norm(predicted_vel, axis=-1)
    
    velocity_result = {
        "ground_truth": ground_truth_vel_magnitude,
        "prediction": predicted_vel_magnitude
    }
    
    n_timesteps = len(ground_truth_vel_magnitude)
    triang = tri.Triangulation(result["node_coords"][0][:, 0]/1e3, result["node_coords"][0][:, 1]/1e3)
    nnode = len(result["node_coords"][0])
    
    return velocity_result, triang, result

def create_output_directory(base_path, member_id):
    """Create dedicated output directory for a member."""
    output_dir = os.path.join(base_path, str(member_id))
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

def extract_timeseries(velocity_result, triang, node_loc):
    points = triang.x, triang.y
    nodes = np.vstack(points).T
    distances = np.linalg.norm(nodes - node_loc, axis=1)
    node_id = np.argmin(distances)
    if node_id is None:
        raise ValueError("Node location not found in triangulation.")
    timeseries = {sim: vel[:, node_id] for sim, vel in velocity_result.items()}
    return timeseries

def get_rupture_time(sliprate_hist, dt, threshold=0.001, unreachable_val=1000):
    n_timesteps, n_nodes = sliprate_hist.shape
    rupture_time = np.full(n_nodes, unreachable_val, dtype=float)
    for it in range(n_timesteps):
        active = (rupture_time == unreachable_val) & (sliprate_hist[it] > threshold)
        rupture_time[active] = it * dt + 1.2
    return rupture_time

def compute_moment(velocity_data, triang, dt, shear_modulus=32e9, slip_threshold=0.01):
    """
    Compute seismic moment M₀ = μ × A × D̄
    
    Parameters:
        velocity_data: slip rate data (m/s) with shape (n_timesteps, n_nodes)
        triang: triangulation object for calculating areas
        dt: time step (s)
        shear_modulus: shear modulus in Pa (default: 30 GPa)
        slip_threshold: minimum slip to consider for rupture area (m)
    
    Returns:
        float: seismic moment in N⋅m
    """
    # Calculate cumulative slip by integrating slip rate over time
    cumulative_slip = np.trapz(velocity_data, dx=dt, axis=0)  # (n_nodes,)
    
    # Identify nodes with significant slip
    active_mask = cumulative_slip > slip_threshold
    
    if np.sum(active_mask) == 0:
        return 0.0
    
    # Calculate rupture area from triangulation
    x, y = triang.x, triang.y
    triangles = triang.triangles
    rupture_area = 0.0
    
    for triangle in triangles:
        # Check if any vertex of triangle has significant slip
        if np.any(active_mask[triangle]):
            # Calculate triangle area using cross product
            p1, p2, p3 = triangle
            x1, y1 = x[p1] * 1000, y[p1] * 1000  # Convert km to m
            x2, y2 = x[p2] * 1000, y[p2] * 1000
            x3, y3 = x[p3] * 1000, y[p3] * 1000
            
            # Triangle area = 0.5 * |cross product|
            area = 0.5 * abs((x2-x1)*(y3-y1) - (x3-x1)*(y2-y1))
            rupture_area += area
    
    # Calculate average slip over active nodes
    average_slip = np.mean(cumulative_slip[active_mask])
    
    # Calculate seismic moment: M₀ = μ × A × D̄
    moment = shear_modulus * rupture_area * average_slip
    
    return moment

def compute_error_metrics(gt, pred, mask_unreachable=True, unreachable_val=1000):
    """
    Compute comprehensive error metrics for rupture time predictions.
    
    Parameters:
        gt: ground truth rupture times
        pred: predicted rupture times
        mask_unreachable: whether to exclude unreachable nodes
        unreachable_val: value indicating unreachable nodes
        
    Returns:
        dict: comprehensive error metrics
    """
    if mask_unreachable:
        valid_mask = (gt < unreachable_val) & (pred < unreachable_val)
        gt_valid = gt[valid_mask]
        pred_valid = pred[valid_mask]
    else:
        gt_valid = gt
        pred_valid = pred
    
    if len(gt_valid) == 0:
        return {"error": "No valid data points for comparison"}
    
    error = pred_valid - gt_valid
    abs_error = np.abs(error)
    
    metrics = {
        "n_valid_nodes": len(gt_valid),
        "n_total_nodes": len(gt),
        "rmse": np.sqrt(mean_squared_error(gt_valid, pred_valid)),
        "mae": mean_absolute_error(gt_valid, pred_valid),
        "mean_error": np.mean(error),
        "std_error": np.std(error),
        "median_error": np.median(error),
        "mean_abs_error": np.mean(abs_error),
        "median_abs_error": np.median(abs_error),
        "max_abs_error": np.max(abs_error),
        "min_abs_error": np.min(abs_error),
        "correlation_coefficient": np.corrcoef(gt_valid, pred_valid)[0, 1],
        "r2_score": r2_score(gt_valid, pred_valid),
        "error_percentiles": {
            "p5": np.percentile(error, 5),
            "p25": np.percentile(error, 25),
            "p75": np.percentile(error, 75),
            "p95": np.percentile(error, 95)
        }
    }
    
    return metrics, error, abs_error, valid_mask

def plot_error_analysis(triang, gt, pred, case_id, case_path, unreachable_val=1000):
    """
    Create comprehensive error analysis plots for rupture time predictions.
    
    Parameters:
        triang: triangulation object
        gt: ground truth rupture times
        pred: predicted rupture times
        case_id: case ID for file naming
        case_path: path to save plots
        unreachable_val: value indicating unreachable nodes
    """
    metrics, error, abs_error, valid_mask = compute_error_metrics(gt, pred, True, unreachable_val)
    
    print(f"\n=== RUPTURE TIME ERROR ANALYSIS (Case {case_id}) ===")
    print(f"Valid nodes: {metrics['n_valid_nodes']}/{metrics['n_total_nodes']}")
    print(f"RMSE: {metrics['rmse']:.4f} s")
    print(f"MAE: {metrics['mae']:.4f} s")
    print(f"Mean Error: {metrics['mean_error']:.4f} s")
    print(f"Std Error: {metrics['std_error']:.4f} s")
    print(f"Correlation: {metrics['correlation_coefficient']:.4f}")
    print(f"R² Score: {metrics['r2_score']:.4f}")
    print(f"Error Range: [{metrics['min_abs_error']:.4f}, {metrics['max_abs_error']:.4f}] s")
    
    fig = plt.figure(figsize=(16, 12))
    
    # 1. Spatial error map
    ax1 = plt.subplot(2, 3, 1)
    error_spatial = np.full(len(gt), np.nan)
    error_spatial[valid_mask] = error
    handle = ax1.tripcolor(triang, error_spatial, cmap='RdBu_r', shading='flat')
    plt.colorbar(handle, ax=ax1, label='Error (s)')
    ax1.set_title('Spatial Error Distribution')
    ax1.set_xlabel('Distance along strike (km)')
    ax1.set_ylabel('Distance along dip (km)')
    ax1.set_aspect('equal')
    
    # 2. Absolute error map
    ax2 = plt.subplot(2, 3, 2)
    abs_error_spatial = np.full(len(gt), np.nan)
    abs_error_spatial[valid_mask] = abs_error
    handle = ax2.tripcolor(triang, abs_error_spatial, cmap='Reds', shading='flat')
    plt.colorbar(handle, ax=ax2, label='Absolute Error (s)')
    ax2.set_title('Spatial Absolute Error Distribution')
    ax2.set_xlabel('Distance along strike (km)')
    ax2.set_ylabel('Distance along dip (km)')
    ax2.set_aspect('equal')
    
    # 3. Error histogram
    ax3 = plt.subplot(2, 3, 3)
    ax3.hist(error, bins=50, alpha=0.7, edgecolor='black', density=True)
    ax3.axvline(0, color='red', linestyle='--', linewidth=2, label='Perfect Prediction')
    ax3.axvline(metrics['mean_error'], color='orange', linestyle='-', linewidth=2, label=f'Mean Error: {metrics["mean_error"]:.3f}s')
    ax3.set_xlabel('Error (s)')
    ax3.set_ylabel('Density')
    ax3.set_title('Error Distribution')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Scatter plot: GT vs Prediction
    ax4 = plt.subplot(2, 3, 4)
    gt_valid = gt[valid_mask]
    pred_valid = pred[valid_mask]
    ax4.scatter(gt_valid, pred_valid, alpha=0.6, s=1)
    min_val, max_val = min(gt_valid.min(), pred_valid.min()), max(gt_valid.max(), pred_valid.max())
    ax4.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    ax4.set_xlabel('Ground Truth Rupture Time (s)')
    ax4.set_ylabel('Predicted Rupture Time (s)')
    ax4.set_title(f'GT vs Prediction (R²={metrics["r2_score"]:.3f})')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. Cumulative error distribution
    ax5 = plt.subplot(2, 3, 5)
    sorted_abs_error = np.sort(abs_error)
    cumulative_prob = np.arange(1, len(sorted_abs_error) + 1) / len(sorted_abs_error)
    ax5.plot(sorted_abs_error, cumulative_prob, linewidth=2)
    ax5.axhline(0.5, color='red', linestyle='--', alpha=0.7, label='50th percentile')
    ax5.axhline(0.9, color='orange', linestyle='--', alpha=0.7, label='90th percentile')
    ax5.axhline(0.95, color='purple', linestyle='--', alpha=0.7, label='95th percentile')
    ax5.set_xlabel('Absolute Error (s)')
    ax5.set_ylabel('Cumulative Probability')
    ax5.set_title('Cumulative Error Distribution')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. Box plot of errors by spatial regions
    ax6 = plt.subplot(2, 3, 6)
    x_coords = triang.x[valid_mask]
    y_coords = triang.y[valid_mask]
    
    # Divide into spatial bins
    x_bins = np.linspace(x_coords.min(), x_coords.max(), 4)
    y_bins = np.linspace(y_coords.min(), y_coords.max(), 4)
    
    box_data = []
    box_labels = []
    for i in range(len(x_bins)-1):
        for j in range(len(y_bins)-1):
            mask_region = ((x_coords >= x_bins[i]) & (x_coords < x_bins[i+1]) & 
                          (y_coords >= y_bins[j]) & (y_coords < y_bins[j+1]))
            if np.sum(mask_region) > 5:  # Only include regions with enough points
                box_data.append(abs_error[mask_region])
                box_labels.append(f'R{i}{j}')
    
    if box_data:
        ax6.boxplot(box_data, labels=box_labels)
        ax6.set_xlabel('Spatial Regions')
        ax6.set_ylabel('Absolute Error (s)')
        ax6.set_title('Error by Spatial Region')
        ax6.grid(True, alpha=0.3)
    else:
        ax6.text(0.5, 0.5, 'Insufficient data\nfor regional analysis', 
                ha='center', va='center', transform=ax6.transAxes)
        ax6.set_title('Error by Spatial Region')
    
    plt.tight_layout()
    plt.savefig(os.path.join(case_path, f"error_analysis_comprehensive_rollout_{case_id}.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save detailed metrics to text file
    with open(os.path.join(case_path, f"error_metrics_rollout_{case_id}.txt"), 'w') as f:
        f.write(f"RUPTURE TIME ERROR ANALYSIS - Case {case_id}\n")
        f.write("="*50 + "\n\n")
        f.write(f"Dataset Information:\n")
        f.write(f"  Total nodes: {metrics['n_total_nodes']}\n")
        f.write(f"  Valid nodes: {metrics['n_valid_nodes']}\n")
        f.write(f"  Coverage: {100*metrics['n_valid_nodes']/metrics['n_total_nodes']:.1f}%\n\n")
        
        f.write(f"Error Metrics:\n")
        f.write(f"  RMSE: {metrics['rmse']:.6f} s\n")
        f.write(f"  MAE: {metrics['mae']:.6f} s\n")
        f.write(f"  Mean Error: {metrics['mean_error']:.6f} s\n")
        f.write(f"  Std Error: {metrics['std_error']:.6f} s\n")
        f.write(f"  Median Error: {metrics['median_error']:.6f} s\n")
        f.write(f"  Max Absolute Error: {metrics['max_abs_error']:.6f} s\n")
        f.write(f"  Min Absolute Error: {metrics['min_abs_error']:.6f} s\n\n")
        
        f.write(f"Statistical Measures:\n")
        f.write(f"  Correlation Coefficient: {metrics['correlation_coefficient']:.6f}\n")
        f.write(f"  R² Score: {metrics['r2_score']:.6f}\n\n")
        
        f.write(f"Error Percentiles:\n")
        for p, val in metrics['error_percentiles'].items():
            f.write(f"  {p}: {val:.6f} s\n")
    
    return metrics

def plot_slip_rate_snapshots(velocity_result, triang, timestep_id, case_id, output_dir, case=4, mode='ground_truth', vmin=0, vmax=10):
    """Plot slip rate snapshots for each simulation at each time step with RMSE."""
    time_s = timestep_id * DT
    time_delay = 1.2 # because the GNS training set starts from 1.2s
    if mode == 'both':
        # Compute snapshot-specific RMSE
        gt_snapshot = velocity_result["ground_truth"][timestep_id]
        pred_snapshot = velocity_result["prediction"][timestep_id]
        snapshot_rmse = np.sqrt(np.mean((pred_snapshot - gt_snapshot)**2))
        snapshot_mae = np.mean(np.abs(pred_snapshot - gt_snapshot))
        
        fig = plt.figure(figsize=(5, 5.5))
        if case == 1:
            fig = plt.figure(figsize=(5, 3.5))
        grid = ImageGrid(fig, 111,
                            nrows_ncols=(2, 1),
                            axes_pad=0.25,
                            share_all=True,
                            cbar_location="right",
                            cbar_mode="single",
                            cbar_size="1.5%",
                            cbar_pad=0.15)
        for j, (sim, vel) in enumerate(velocity_result.items()):
            handle = grid[j].tripcolor(triang, vel[timestep_id], vmax=vmax, vmin=vmin)
            cbar = fig.colorbar(handle, cax=grid.cbar_axes[0])
            cbar.set_label("Slip Rate (m/s)")
            grid[j].set_title(sim, fontweight='bold')
        
        # Add metrics text
        #metrics_text = f"t = {time_s+time_delay:.1f}s\nRMSE: {snapshot_rmse:.4f} m/s\nMAE: {snapshot_mae:.4f} m/s"
        metrics_text = f"t = {time_s+time_delay:.1f}s"

        grid[1].text(0.02, 0.02, metrics_text, transform=grid[1].transAxes, fontsize=9,
                    verticalalignment='bottom', horizontalalignment='left',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9))
        
        label = grid[0].set_ylabel("Distance along dip (km)")
        label.set_position((-0.0, -0.4))
        grid[1].set_xlabel("Distance along strike (km)")
    else:
        fig = plt.figure(figsize=(6, 3))
        grid = ImageGrid(fig, 111, nrows_ncols=(1, 1))
        handle = grid[0].tripcolor(triang, velocity_result[mode][timestep_id], vmax=vmax, vmin=vmin)
        grid[0].set_xticks([])
        grid[0].set_yticks([])
        
        # Add time stamp
        grid[0].text(0.02, 0.98, f"t = {time_s+time_delay:.1f}s", transform=grid[0].transAxes, fontsize=10,
                    verticalalignment='top', horizontalalignment='left',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9))
    
    fig.savefig(os.path.join(output_dir, f"a.Sliprate_snapshot_time_{mode}_{timestep_id}_rollout_{case_id}.png"), dpi=600)
    plt.close(fig)

def plot_combined_rpt_contours(triang, rpt_gt, rpt_pred, filename, case=4, asp_rect=None, 
                              global_rmse=None, rpt_rmse=None, moment_gt=None, moment_pred=None):
    """Plot combined rupture time contours for ground truth and prediction with metrics."""
    fig, ax = plt.subplots(figsize=(5, 4))
    if case == 1:
        fig, ax = plt.subplots(figsize=(8, 4))
    
    mask_gt = np.ma.masked_where(rpt_gt >= 999, rpt_gt)
    mask_pred = np.ma.masked_where(rpt_pred >= 999, rpt_pred)
    
    levels = np.arange(0, 15, 0.5)
    mask_gt = mask_gt.filled(0)
    cs_gt = ax.tricontour(triang, mask_gt, levels=levels, colors='black', linewidths=1.0)
    contour_labels = ax.clabel(cs_gt, fmt=lambda x: f'{x:.1f}', inline=True, inline_spacing=20, fontsize=7)
    for label in contour_labels:
        label.set_position(label.get_position() + np.array([0, 0.3]))
    
    mask_pred = mask_pred.filled(0)
    cs_pred = ax.tricontour(triang, mask_pred, levels=levels, colors='red', linestyles='--', linewidths=1.0)
    contour_labels = ax.clabel(cs_pred, fmt=lambda x: f'{x:.1f}', inline=True, inline_spacing=10, fontsize=7)
    for label in contour_labels:
        label.set_position(label.get_position() + np.array([0, -0.3]))
    
    if asp_rect is not None:
        for rect in asp_rect:
            ax.add_patch(rect)
    
    # Add legend in upper left
    legend_elements = [
        plt.Line2D([0], [0], color='black', linewidth=1.5, label='Ground Truth'),
        plt.Line2D([0], [0], color='red', linewidth=1.5, linestyle='--', label='Prediction')
    ]
    legend = ax.legend(handles=legend_elements, loc='upper left', fontsize=7, 
                      bbox_to_anchor=(0.02, 0.98), framealpha=1.0, 
                      facecolor='white', edgecolor='black')
    legend.set_zorder(1000)
    
    # Add metrics box in upper right
    if global_rmse is not None or rpt_rmse is not None or (moment_gt is not None and moment_pred is not None):
        metrics_lines = []
        if global_rmse is not None:
            metrics_lines.append(f"SR RMSE: {global_rmse:.2f} m/s")
        if rpt_rmse is not None:
            metrics_lines.append(f"RT RMSE: {rpt_rmse:.2f} s")
        
        if moment_gt is not None and moment_pred is not None:
            mag_gt = (2.0/3.0) * (np.log10(moment_gt) - 9.1)
            mag_pred = (2.0/3.0) * (np.log10(moment_pred) - 9.1)
            metrics_lines.append(f"GT Mw: {mag_gt:.2f}")
            metrics_lines.append(f"Pred Mw: {mag_pred:.2f}")
        
        if metrics_lines:
            metrics_text = '\n'.join(metrics_lines)
            ax.text(0.98, 0.98, metrics_text, transform=ax.transAxes, fontsize=7,
                   verticalalignment='top', horizontalalignment='right',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=1.0, 
                            edgecolor='black', linewidth=1.0),
                   zorder=1000)
    
    ax.grid(True)
    ax.set_xlabel("Distance along strike (km)")
    ax.set_ylabel("Distance along dip (km)")
    ax.set_aspect('equal', adjustable='box')
    fig.tight_layout()
    fig.savefig(filename, dpi=600)
    plt.close(fig)

def plot_slip_rate_time_series(velocity_result, triang, node_loc, dt, case_id, output_dir):
    """Extract and plot the slip rate time series for a specific node location with RMSE."""
    timeseries = extract_timeseries(velocity_result, triang, node_loc)
    
    # Compute node-specific RMSE
    gt_series = timeseries["ground_truth"]
    pred_series = timeseries["prediction"]
    node_rmse = np.sqrt(np.mean((pred_series - gt_series)**2))
    node_mae = np.mean(np.abs(pred_series - gt_series))
    node_corr = np.corrcoef(gt_series, pred_series)[0,1] if np.std(gt_series) > 1e-8 and np.std(pred_series) > 1e-8 else 0.0
    
    fig, ax = plt.subplots(figsize=(4.2, 4))
    for sim, vel in timeseries.items():
        ax.plot(np.arange(len(vel)) * dt, vel, label=sim, linewidth=2)
    
    # Add metrics text box
    metrics_text = f"RMSE: {node_rmse:.4f} m/s\nMAE: {node_mae:.4f} m/s\nCorr: {node_corr:.3f}"
    ax.text(0.02, 0.98, metrics_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', horizontalalignment='left',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.8))
    
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Slip Rate (m/s)")
    ax.set_title(f"Node Location {node_loc}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"b.Sliprate_time_series_node_{node_loc[0]}_{node_loc[1]}_rollout_{case_id}.png"), dpi=600)
    plt.close(fig)

def compute_global_timeseries_metrics(velocity_result, triang):
    """Compute comprehensive time series metrics across all stations."""
    gt_vel = velocity_result["ground_truth"]  # shape: (n_timesteps, n_nodes)
    pred_vel = velocity_result["prediction"]
    
    n_timesteps, n_nodes = gt_vel.shape
    
    # Compute per-timestep correlations safely
    timestep_corr = []
    debug_stats = []
    for t in range(n_timesteps):
        gt_t, pred_t = gt_vel[t], pred_vel[t]
        gt_std, pred_std = np.std(gt_t), np.std(pred_t)
        gt_max, pred_max = np.max(gt_t), np.max(pred_t)
        
        if gt_std > 1e-8 and pred_std > 1e-8:
            corr = np.corrcoef(gt_t, pred_t)[0,1]
            timestep_corr.append(corr if not np.isnan(corr) else 0.0)
        else:
            timestep_corr.append(0.0)
        
        # Store debug info for problematic timesteps
        if t < 10 or t % 50 == 0:  # Sample key timesteps
            debug_stats.append({
                'timestep': t,
                'time_s': t * DT,
                'correlation': timestep_corr[-1],
                'gt_max': gt_max,
                'pred_max': pred_max,
                'gt_std': gt_std,
                'pred_std': pred_std
            })
    
    # Compute per-node correlations safely
    node_corr = []
    for n in range(n_nodes):
        gt_n, pred_n = gt_vel[:, n], pred_vel[:, n]
        if np.std(gt_n) > 1e-8 and np.std(pred_n) > 1e-8:
            corr = np.corrcoef(gt_n, pred_n)[0,1]
            node_corr.append(corr if not np.isnan(corr) else 0.0)
        else:
            node_corr.append(0.0)
    
    # Compute timing-robust metrics
    # Energy ~ slip_rate^2 (kinetic energy density)
    energy_gt = np.sum(gt_vel**2, axis=1)      # Total energy per timestep
    energy_pred = np.sum(pred_vel**2, axis=1)
    energy_correlation = np.corrcoef(energy_gt, energy_pred)[0,1] if np.std(energy_gt) > 1e-8 and np.std(energy_pred) > 1e-8 else 0.0
    
    # Also compute moment rate (sum of slip rates - more interpretable)
    moment_gt = np.sum(gt_vel, axis=1)
    moment_pred = np.sum(pred_vel, axis=1) 
    moment_correlation = np.corrcoef(moment_gt, moment_pred)[0,1] if np.std(moment_gt) > 1e-8 and np.std(moment_pred) > 1e-8 else 0.0
    
    # Peak timing comparison (less sensitive to minor delays)
    active_threshold = 0.1  # m/s threshold for "active" slip
    gt_active = (gt_vel > active_threshold).astype(int)
    pred_active = (pred_vel > active_threshold).astype(int)
    activity_correlation = []
    
    for t in range(n_timesteps):
        gt_act, pred_act = gt_active[t], pred_active[t]
        if np.sum(gt_act) > 0 and np.sum(pred_act) > 0:
            # Jaccard similarity for active regions
            intersection = np.sum(gt_act * pred_act)
            union = np.sum((gt_act + pred_act) > 0)
            activity_correlation.append(intersection / union if union > 0 else 0.0)
        else:
            activity_correlation.append(1.0 if np.sum(gt_act) == 0 and np.sum(pred_act) == 0 else 0.0)
    
    metrics = {
        # Per-timestep metrics
        "timestep_rmse": np.sqrt(np.mean((pred_vel - gt_vel)**2, axis=1)),
        "timestep_mae": np.mean(np.abs(pred_vel - gt_vel), axis=1),
        "timestep_correlation": np.array(timestep_corr),
        "timestep_energy_correlation": energy_correlation,
        "timestep_activity_similarity": np.array(activity_correlation),
        
        # Per-node metrics  
        "node_rmse": np.sqrt(np.mean((pred_vel - gt_vel)**2, axis=0)),
        "node_mae": np.mean(np.abs(pred_vel - gt_vel), axis=0),
        "node_correlation": np.array(node_corr),
        
        # Global metrics
        "global_rmse": np.sqrt(np.mean((pred_vel - gt_vel)**2)),
        "global_mae": np.mean(np.abs(pred_vel - gt_vel)),
        "global_correlation": np.corrcoef(gt_vel.flatten(), pred_vel.flatten())[0,1],
        
        # Timing-robust metrics
        "energy_correlation": energy_correlation,
        "moment_correlation": moment_correlation,
        "mean_activity_similarity": np.mean(activity_correlation),
        
        # Debug info
        "debug_stats": debug_stats
    }
    
    return metrics

def plot_global_timeseries_analysis(velocity_result, triang, case_id, output_dir):
    """Create comprehensive all-station time series comparison plots."""
    metrics = compute_global_timeseries_metrics(velocity_result, triang)
    plot_global_timeseries_analysis_with_metrics(metrics, velocity_result, triang, case_id, output_dir)
    return metrics

def plot_global_timeseries_analysis_with_metrics(metrics, velocity_result, triang, case_id, output_dir):
    """Create comprehensive all-station time series comparison plots using precomputed metrics."""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. Error evolution over time
    time_axis = np.arange(len(metrics["timestep_rmse"])) * DT
    axes[0,0].plot(time_axis, metrics["timestep_rmse"], 'b-', label='RMSE', linewidth=2)
    axes[0,0].plot(time_axis, metrics["timestep_mae"], 'r-', label='MAE', linewidth=2) 
    axes[0,0].set_xlabel('Time (s)')
    axes[0,0].set_ylabel('Error (m/s)')
    axes[0,0].set_title('Global Error Evolution')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    
    # 2. Correlation evolution over time (multiple metrics)
    axes[0,1].plot(time_axis, metrics["timestep_correlation"], 'g-', linewidth=2, label='Spatial Correlation')
    axes[0,1].plot(time_axis, metrics["timestep_activity_similarity"], 'b-', linewidth=2, label='Activity Similarity')
    axes[0,1].axhline(y=metrics["energy_correlation"], color='r', linestyle='-', alpha=0.7, 
                     label=f'Energy: {metrics["energy_correlation"]:.3f}')
    axes[0,1].axhline(y=metrics["moment_correlation"], color='orange', linestyle='--', alpha=0.7, 
                     label=f'Moment: {metrics["moment_correlation"]:.3f}')
    axes[0,1].set_xlabel('Time (s)')
    axes[0,1].set_ylabel('Correlation/Similarity')
    axes[0,1].set_title('Temporal Correlation Metrics')
    axes[0,1].set_ylim([-1, 1])
    axes[0,1].axhline(y=0, color='k', linestyle='--', alpha=0.5)
    axes[0,1].legend(fontsize=7)
    axes[0,1].grid(True, alpha=0.3)
    
    # 3. Spatial error distribution
    im = axes[0,2].tripcolor(triang, metrics["node_rmse"], cmap='Reds', shading='flat')
    plt.colorbar(im, ax=axes[0,2], label='RMSE (m/s)')
    axes[0,2].set_title('Spatial RMSE Distribution')
    axes[0,2].set_xlabel('Distance along strike (km)')
    axes[0,2].set_ylabel('Distance along dip (km)')
    axes[0,2].set_aspect('equal')
    
    # 4. Spatial correlation distribution  
    im = axes[1,0].tripcolor(triang, metrics["node_correlation"], cmap='RdBu_r', 
                            vmin=-1, vmax=1, shading='flat')
    plt.colorbar(im, ax=axes[1,0], label='Correlation')
    axes[1,0].set_title('Spatial Correlation Distribution')
    axes[1,0].set_xlabel('Distance along strike (km)')
    axes[1,0].set_ylabel('Distance along dip (km)')
    axes[1,0].set_aspect('equal')
    
    # 5. Error histogram across all stations
    all_errors = (velocity_result["prediction"] - velocity_result["ground_truth"]).flatten()
    axes[1,1].hist(all_errors, bins=100, alpha=0.7, density=True, edgecolor='black', linewidth=0.5)
    axes[1,1].axvline(x=0, color='red', linestyle='--', linewidth=2, label='Perfect Prediction')
    axes[1,1].axvline(x=np.mean(all_errors), color='orange', linestyle='-', linewidth=2, 
                     label=f'Mean: {np.mean(all_errors):.3f}')
    axes[1,1].set_xlabel('Error (m/s)')
    axes[1,1].set_ylabel('Density')
    axes[1,1].set_title('Global Error Distribution')
    axes[1,1].legend()
    axes[1,1].grid(True, alpha=0.3)
    
    # 6. Peak slip rate comparison
    peak_gt = np.max(velocity_result["ground_truth"], axis=0)
    peak_pred = np.max(velocity_result["prediction"], axis=0)
    axes[1,2].scatter(peak_gt, peak_pred, alpha=0.6, s=10)
    max_val = max(peak_gt.max(), peak_pred.max())
    axes[1,2].plot([0, max_val], [0, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    axes[1,2].set_xlabel('GT Peak Slip Rate (m/s)')
    axes[1,2].set_ylabel('Predicted Peak Slip Rate (m/s)')
    axes[1,2].set_title('Peak Slip Rate Comparison')
    axes[1,2].legend()
    axes[1,2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"c.global_timeseries_analysis_rollout_{case_id}.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save comprehensive metrics
    with open(os.path.join(output_dir, f"global_timeseries_metrics_rollout_{case_id}.txt"), 'w') as f:
        f.write(f"GLOBAL TIME SERIES ANALYSIS - Case {case_id}\n")
        f.write("="*50 + "\n\n")
        f.write(f"Global Metrics:\n")
        f.write(f"  Global RMSE: {metrics['global_rmse']:.6f} m/s\n")
        f.write(f"  Global MAE: {metrics['global_mae']:.6f} m/s\n")
        f.write(f"  Global Correlation: {metrics['global_correlation']:.6f}\n\n")
        
        f.write(f"Temporal Statistics:\n")
        f.write(f"  Mean timestep RMSE: {np.mean(metrics['timestep_rmse']):.6f} m/s\n")
        f.write(f"  Max timestep RMSE: {np.max(metrics['timestep_rmse']):.6f} m/s\n")
        f.write(f"  Mean spatial correlation: {np.mean(metrics['timestep_correlation']):.6f}\n")
        f.write(f"  Min spatial correlation: {np.min(metrics['timestep_correlation']):.6f}\n")
        f.write(f"  Energy correlation: {metrics['energy_correlation']:.6f}\n")
        f.write(f"  Moment rate correlation: {metrics['moment_correlation']:.6f}\n")
        f.write(f"  Mean activity similarity: {metrics['mean_activity_similarity']:.6f}\n\n")
        
        f.write(f"Spatial Statistics:\n")
        f.write(f"  Mean node RMSE: {np.mean(metrics['node_rmse']):.6f} m/s\n")
        f.write(f"  Max node RMSE: {np.max(metrics['node_rmse']):.6f} m/s\n")
        f.write(f"  Mean node correlation: {np.mean(metrics['node_correlation']):.6f}\n")
        f.write(f"  Min node correlation: {np.min(metrics['node_correlation']):.6f}\n\n")
        
        # Add debug information for temporal correlation issues
        if 'debug_stats' in metrics and len(metrics['debug_stats']) > 0:
            f.write(f"Temporal Correlation Debug (Key Timesteps):\n")
            f.write(f"{'Step':<6} {'Time(s)':<8} {'Correlation':<12} {'GT_Max':<10} {'Pred_Max':<10} {'GT_Std':<10} {'Pred_Std':<10}\n")
            f.write("-" * 70 + "\n")
            for stat in metrics['debug_stats']:
                f.write(f"{stat['timestep']:<6} {stat['time_s']:<8.3f} {stat['correlation']:<12.6f} "
                       f"{stat['gt_max']:<10.4f} {stat['pred_max']:<10.4f} "
                       f"{stat['gt_std']:<10.6f} {stat['pred_std']:<10.6f}\n")
    
    return metrics

def create_metrics_table_and_csv(global_metrics, error_metrics, case_id, output_dir):
    """Create comprehensive metrics table figure and CSV file."""
    import pandas as pd
    
    # Organize all metrics into categories
    metrics_data = {
        'Category': [],
        'Metric': [],
        'Value': [],
        'Unit': [],
        'Description': []
    }
    
    # Global Time Series Metrics
    global_data = [
        ('Global', 'Time Series RMSE', global_metrics['global_rmse'], 'm/s', 'Root mean square error across all nodes and timesteps'),
        ('Global', 'Time Series MAE', global_metrics['global_mae'], 'm/s', 'Mean absolute error across all nodes and timesteps'),
        ('Global', 'Time Series Correlation', global_metrics['global_correlation'], '-', 'Pearson correlation between all GT and predicted values'),
        ('Global', 'Energy Correlation', global_metrics['energy_correlation'], '-', 'Correlation of energy release timing (∑v²)'),
        ('Global', 'Moment Rate Correlation', global_metrics['moment_correlation'], '-', 'Correlation of moment release timing (∑v)'),
        ('Global', 'Activity Similarity', global_metrics['mean_activity_similarity'], '-', 'Mean Jaccard similarity of active regions')
    ]
    
    # Temporal Statistics
    temporal_data = [
        ('Temporal', 'Mean Timestep RMSE', np.mean(global_metrics['timestep_rmse']), 'm/s', 'Average RMSE across all timesteps'),
        ('Temporal', 'Max Timestep RMSE', np.max(global_metrics['timestep_rmse']), 'm/s', 'Maximum RMSE at any timestep'),
        ('Temporal', 'Mean Spatial Correlation', np.mean(global_metrics['timestep_correlation']), '-', 'Average spatial correlation over time'),
        ('Temporal', 'Min Spatial Correlation', np.min(global_metrics['timestep_correlation']), '-', 'Minimum spatial correlation over time')
    ]
    
    # Spatial Statistics  
    spatial_data = [
        ('Spatial', 'Mean Node RMSE', np.mean(global_metrics['node_rmse']), 'm/s', 'Average RMSE across all nodes'),
        ('Spatial', 'Max Node RMSE', np.max(global_metrics['node_rmse']), 'm/s', 'Maximum RMSE at any node'),
        ('Spatial', 'Mean Node Correlation', np.mean(global_metrics['node_correlation']), '-', 'Average correlation across all nodes'),
        ('Spatial', 'Min Node Correlation', np.min(global_metrics['node_correlation']), '-', 'Minimum correlation at any node')
    ]
    
    # Rupture Time Metrics
    rupture_data = [
        ('Rupture Time', 'RMSE', error_metrics['rmse'], 's', 'Root mean square error of rupture time prediction'),
        ('Rupture Time', 'MAE', error_metrics['mae'], 's', 'Mean absolute error of rupture time prediction'),
        ('Rupture Time', 'Correlation', error_metrics['correlation_coefficient'], '-', 'Correlation between GT and predicted rupture times'),
        ('Rupture Time', 'R² Score', error_metrics['r2_score'], '-', 'Coefficient of determination for rupture time'),
        ('Rupture Time', 'Coverage', error_metrics['n_valid_nodes']/error_metrics['n_total_nodes']*100, '%', 'Percentage of nodes with valid rupture times')
    ]
    
    # Combine all data
    all_data = global_data + temporal_data + spatial_data + rupture_data
    
    for category, metric, value, unit, description in all_data:
        metrics_data['Category'].append(category)
        metrics_data['Metric'].append(metric)
        metrics_data['Value'].append(value)
        metrics_data['Unit'].append(unit)
        metrics_data['Description'].append(description)
    
    # Create DataFrame
    df = pd.DataFrame(metrics_data)
    
    # Save to CSV
    csv_filename = os.path.join(output_dir, f"comprehensive_metrics_rollout_{case_id}.csv")
    df.to_csv(csv_filename, index=False)
    
    # Create figure with table
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.axis('tight')
    ax.axis('off')
    
    # Prepare table data for display (round values for readability)
    table_data = []
    for i, row in df.iterrows():
        value = row['Value']
        if isinstance(value, (int, float)):
            if abs(value) < 0.001:
                value_str = f"{value:.2e}"
            elif abs(value) < 0.01:
                value_str = f"{value:.4f}"
            else:
                value_str = f"{value:.3f}"
        else:
            value_str = str(value)
        
        table_data.append([row['Category'], row['Metric'], value_str, row['Unit']])
    
    # Create table
    table = ax.table(cellText=table_data,
                    colLabels=['Category', 'Metric', 'Value', 'Unit'],
                    cellLoc='left',
                    loc='center',
                    colWidths=[0.20, 0.35, 0.15, 0.10])
    
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.8)
    
    # Color code by category
    colors = {'Global': '#E8F4FF', 'Temporal': '#FFE8E8', 'Spatial': '#E8FFE8', 'Rupture Time': '#FFF8E8'}
    for i, (category, _, _, _) in enumerate(table_data):
        for j in range(4):
            table[(i+1, j)].set_facecolor(colors.get(category, '#FFFFFF'))
    
    # Header styling
    for j in range(4):
        table[(0, j)].set_facecolor('#D0D0D0')
        table[(0, j)].set_text_props(weight='bold')
    
    plt.title(f'Comprehensive Performance Metrics - Case {case_id}', fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"d.comprehensive_metrics_table_rollout_{case_id}.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    return csv_filename, df

def process_member(member_id, case=4):
    """Process a single member and generate all outputs."""
    # Get case configuration
    config = get_case_config(case)
    case_path = config["case_path"]
    test_metadata_path = config["test_metadata"]
    
    # Create output directory
    output_dir = create_output_directory(case_path, member_id)
    
    try:
        # Load data
        velocity_result, triang, result = load_rollout_data(case_path, member_id)
        is_asperity, asp_rect = load_asperity_info(test_metadata_path, member_id)
        
        # Compute rupture times
        rpt_ground_truth = get_rupture_time(velocity_result["ground_truth"], DT, SLIPRATE_THRESHOLD, 1000)
        rpt_predicted = get_rupture_time(velocity_result["prediction"], DT, SLIPRATE_THRESHOLD, 1000)
        
        # Comprehensive error analysis
        error_metrics = plot_error_analysis(triang, rpt_ground_truth, rpt_predicted, member_id, output_dir)
        
        # Global time series analysis (compute metrics first)
        global_metrics = compute_global_timeseries_metrics(velocity_result, triang)
        
        # Compute earthquake moments
        moment_gt = compute_moment(velocity_result["ground_truth"], triang, DT)
        moment_pred = compute_moment(velocity_result["prediction"], triang, DT)
        
        # Rupture time comparison plot with metrics overlay
        plot_combined_rpt_contours(
            triang,
            rpt_ground_truth,
            rpt_predicted,
            os.path.join(output_dir, f"a.rupture_time_comparison_rollout_{member_id}.png"),
            case=case,
            asp_rect=asp_rect if is_asperity else None,
            global_rmse=global_metrics['global_rmse'],
            rpt_rmse=error_metrics['rmse'],
            moment_gt=moment_gt,
            moment_pred=moment_pred
        )
        
        # Slip rate snapshots
        for mode in ['ground_truth', 'both']:
            for time_step in [0, 1, 2, 3, 4, 5]:
                timestep_id = int(time_step / DT)
                plot_slip_rate_snapshots(velocity_result, triang, timestep_id, member_id, output_dir, case=case, mode=mode)
        
        # Time series plots at specific node locations
        node_locations = [(-1, -7), (3, -3), (7, -7), (-1, -3), (3, -7), (7, -3)]
        for node_loc in node_locations:
            plot_slip_rate_time_series(velocity_result, triang, node_loc, DT, member_id, output_dir)
        
        # Generate global time series analysis plots (metrics already computed)
        plot_global_timeseries_analysis_with_metrics(global_metrics, velocity_result, triang, member_id, output_dir)
        
        # Create comprehensive metrics table and CSV
        csv_filename, metrics_df = create_metrics_table_and_csv(global_metrics, error_metrics, member_id, output_dir)
        
        # Summary statistics
        mag_gt = (2.0/3.0) * (np.log10(moment_gt) - 9.1) if moment_gt > 0 else 0.0
        mag_pred = (2.0/3.0) * (np.log10(moment_pred) - 9.1) if moment_pred > 0 else 0.0
        
        print(f"\nMember {member_id} Summary:")
        print(f"  Global Time Series RMSE: {global_metrics['global_rmse']:.6f} m/s")
        print(f"  Rupture Time RMSE: {error_metrics['rmse']:.6f} s")
        print(f"  Energy Correlation: {global_metrics['energy_correlation']:.6f}")
        print(f"  Activity Similarity: {global_metrics['mean_activity_similarity']:.6f}")
        print(f"  GT Moment: {moment_gt:.2e} N⋅m (Mw {mag_gt:.2f})")
        print(f"  Pred Moment: {moment_pred:.2e} N⋅m (Mw {mag_pred:.2f})")
        print(f"  Magnitude Error: {abs(mag_pred - mag_gt):.3f}")
        print(f"  Max predicted rupture time: {np.max(rpt_predicted):.3f} s")
        print(f"  Ground truth slip max: {np.max(velocity_result['ground_truth']):.3f} m/s")
        print(f"  Prediction slip max: {np.max(velocity_result['prediction']):.3f} m/s")
        print(f"  Results saved to: {output_dir}")
        print(f"  Metrics CSV: {csv_filename}")
        
        return True
        
    except Exception as e:
        print(f"Error processing member {member_id}: {str(e)}")
        return False

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Enhanced rupture dynamics visualization with comprehensive error analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python plot.rupture.dynamics_mod.py 5        # Process member 5
  python plot.rupture.dynamics_mod.py --all    # Process all available members
  python plot.rupture.dynamics_mod.py --case 2 --all  # Process all available members for case 2
  python plot.rupture.dynamics_mod.py --all --max-members 10  # Process members 0-10 only
        """
    )
    
    parser.add_argument('member_id', nargs='?', type=int, help='Member ID to process')
    parser.add_argument('--all', action='store_true', help='Process all available members')
    parser.add_argument('--case', type=int, default=4, choices=range(20), 
                       help='Case number (0-20, default: 4)')
    parser.add_argument('--max-members', type=int, default=None, 
                       help='Maximum member ID when using --all (default: auto-detect from available pkl files)')
    
    return parser.parse_args()

def main():
    """Main function to handle argument parsing and member processing."""
    args = parse_arguments()
    
    # Get case configuration to detect available members
    config = get_case_config(args.case)
    case_path = config["case_path"]
    
    # Detect available members
    available_members = detect_available_members(case_path)
    if not available_members:
        print(f"Error: No rollout pkl files found in {case_path}")
        sys.exit(1)
    
    max_available = max(available_members)
    min_available = min(available_members)
    
    print(f"Found {len(available_members)} pkl files: members {min_available}-{max_available}")
    
    # Validate arguments
    if args.all and args.member_id is not None:
        print("Error: Cannot specify both --all and a specific member_id")
        sys.exit(1)
    
    if not args.all and args.member_id is None:
        print("Error: Must specify either a member_id or --all")
        sys.exit(1)
    
    if args.member_id is not None and args.member_id not in available_members:
        print(f"Error: member_id {args.member_id} not found. Available members: {available_members}")
        sys.exit(1)
    
    print(f"Starting rupture dynamics analysis for case {args.case}")
    
    success_count = 0
    total_count = 0
    
    if args.all:
        # Determine which members to process
        if args.max_members is not None:
            members_to_process = [m for m in available_members if m <= args.max_members]
        else:
            members_to_process = available_members
        
        print(f"Processing {len(members_to_process)} members: {members_to_process}")
        
        for member_id in members_to_process:
            print(f"\n--- Processing Member {member_id} ---")
            total_count += 1
            if process_member(member_id, args.case):
                success_count += 1
    else:
        print(f"Processing member {args.member_id}")
        total_count = 1
        if process_member(args.member_id, args.case):
            success_count = 1
    
    print(f"\n=== PROCESSING COMPLETE ===")
    print(f"Successfully processed: {success_count}/{total_count} members")
    if success_count < total_count:
        print(f"Failed to process: {total_count - success_count} members")

if __name__ == "__main__":
    main()