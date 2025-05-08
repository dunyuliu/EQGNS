import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import BoundaryNorm, ListedColormap
import numpy as np
import os
from collections import defaultdict

plt.rcParams.update({'font.size': 14})

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
    
cmap = ListedColormap(plt.cm.viridis(np.linspace(0, 1, 20)))
bounds = np.linspace(-0.1, 1.1, 20)
norm = BoundaryNorm(bounds, cmap.N)

# === CONFIGURATION ===
# Folder or file pattern containing your datasets

dataset_root_path = "case4.200m.multi.stress/dataset"
dataset_filenames = {'train': 'case4.200m.multi.stress.train.metadata.json',
                    'valid': 'case4.200m.multi.stress.valid.metadata.json',
                    'test': 'case4.200m.multi.stress.test.metadata.json'}

# Function to convert stress values to color
#def stress_to_color(stress1, stress2):
#    avg_stress = (stress1 + stress2) / 2
#    cmap = plt.cm.viridis
#    return cmap(avg_stress)

# Loop over each JSON file
for i, (key, json_file) in enumerate(dataset_filenames.items()):
    json_file = os.path.join(dataset_root_path, json_file)

    with open(json_file) as f:
        data = json.load(f)

        # Group items by hypocenter location
    grouped_by_hypo = defaultdict(list)

    for item in data:
        hypo = tuple(item["hypocenter_location_km"])
        grouped_by_hypo[hypo].append(item)

    for hypo_loc, items in grouped_by_hypo.items():
        fig, ax = plt.subplots(figsize=(10, 8))
         # Plot the common hypocenter
        ax.plot(hypo_loc[0], hypo_loc[1], marker='*', color='red', markersize=25)

        for item in items:
            # Plot asperity
            asp_x, asp_y, half_size, stress1, stress2 = item["asperity_location_km"]
            full_size = 2 * half_size
            color = stress_to_color(stress2)

            rect = patches.Rectangle(
                (asp_x - half_size, asp_y - half_size),
                full_size, full_size,
                linewidth=3,
                edgecolor=color,
                facecolor='none'
            )
            ax.add_patch(rect)

        # Final plot setup
        ax.set_xlabel("Strike (km)")
        ax.set_ylabel("Dip (km)")
        #ax.set_title("2D Fault Plot: Hypocenters & Asperities from Multiple Datasets")
        ax.set_aspect("equal")
        ax.grid(True)
        ax.set_xlim(-9, 9)
        ax.set_ylim(-10, 0)
        # Add colorbar
        #sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        #fig.colorbar(sm, ax=ax, orientation="horizontal", label="Normalized Asperity Stress")

        legend_patches = [
            patches.Patch(color=colors[i], label=f'{stress_bins[i]:.2f}')
            for i in range(len(stress_bins))
        ]
        ax.legend(handles=legend_patches, title="Normalized Stress", fontsize=9, title_fontsize=10, loc='best')
        plt.tight_layout()
        tag = f"x{hypo_loc[0]}_y{hypo_loc[1]}".replace('.', 'p')
        plt.savefig(f"0.production_figures/case4.200m.multi.stress.{key}.{tag}.dataset.png", dpi=600)
        plt.close(fig)