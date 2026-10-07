#! /usr/bin/env python
# -*- coding: utf-8 -*-
import json
import matplotlib.pyplot as plt
import numpy as np

font = {'family': 'serif',
        'weight': 'bold',
        'size': 10}
plt.rc('font', **font)
plt.rcParams['axes.labelweight'] = font['weight']     # Ensures bold axis labels
plt.rcParams['axes.labelsize'] = font['size']          # Ensures correct font size for xlabel/ylabel

metadata_root_path = "../results/case3.200m.homo.a.Vw/dataset/"

# File paths for the JSON files
train_file = metadata_root_path+"case3_200m_train.npz.metadata.json"
valid_file = metadata_root_path+"case3_200m_valid.npz.metadata.json"
test_file = metadata_root_path+"case3_200m_test.npz.metadata.json"

# Load data from JSON files
def load_hypocenters(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return [(entry['hypocenter_location_km'][0], entry['hypocenter_location_km'][1]) for entry in data]  # Extract x and y from 'hypocenter_location_km'

train_hypocenters = load_hypocenters(train_file)
valid_hypocenters = load_hypocenters(valid_file)
test_hypocenters = load_hypocenters(test_file)

# Extract coordinates
def extract_coordinates(hypocenters):
    x, y = zip(*hypocenters)
    return x, y

train_x, train_y = extract_coordinates(train_hypocenters)
valid_x, valid_y = extract_coordinates(valid_hypocenters)
test_x, test_y = extract_coordinates(test_hypocenters)

# Define the grid spacing
dx = dy = 0.2

# Generate x and y coordinates
x = np.arange(-9, 9 + dx, dx)
y = np.arange(-10, 0 + dy, dy)

# Create the meshgrid
X, Y = np.meshgrid(x, y)

# Plotting
plt.figure(figsize=(6, 4))  # Adjust figure size to be smaller

ax = plt.gca()  # Get current axis
plt.plot(X, Y, marker='.', color='k', linestyle='none', markersize=2)
# Plot each dataset with larger dots and adjusted alpha
plt.scatter(train_x, train_y, c='blue', label='Train', alpha=0.8, s=20)
plt.scatter(valid_x, valid_y, c='green', label='Valid', alpha=0.8, s=20)
plt.scatter(test_x, test_y, c='red', label='Test', alpha=0.8, s=20)

# Labels and legend with larger font sizes
plt.xlabel('Distance along strike (km)', fontsize=12)
plt.ylabel('Distance along dip (km)', fontsize=12)
#plt.title('Hypocenter Locations on Fault Plane (2D)', fontsize=14)
plt.legend(fontsize=10, loc="best", markerscale=1.5)  # Adjust legend font size and marker scale

# Adjust font weight for better visibility
plt.xticks(fontsize=10, weight='bold')
plt.yticks(fontsize=10, weight='bold')

plt.axis('equal')

# Optional: set font size for ticks
ax.tick_params(labelsize=10)

# Save the plot to the root path
output_path = metadata_root_path + "hypocenter_locations_plot.png"
plt.savefig(output_path, dpi=600, bbox_inches='tight')
plt.tight_layout()
plt.show()
