import pickle
from absl import app
from absl import flags
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.tri as tri
from mpl_toolkits.axes_grid1 import ImageGrid
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib import animation
import json, os

flags.DEFINE_string("rollout_dir", None, help="Directory where rollout.pkl are located")
flags.DEFINE_string("rollout_name", None, help="Name of rollout `.pkl` file")
flags.DEFINE_integer("step_stride", 3, help="Stride of steps to skip.")
FLAGS = flags.FLAGS

def render_gif_animation():

    rollout_path = f"{FLAGS.rollout_dir}/{FLAGS.rollout_name}.pkl"
    animation_filename = f"{FLAGS.rollout_dir}/{FLAGS.rollout_name}.gif"
    
    # if testset metadata exists, use the info for better rendering.
    testset_metadata_name = f"{FLAGS.rollout_dir}/testset_metadata.json"
    metadata_file_exists = os.path.exists(testset_metadata_name)

    if metadata_file_exists == True:
        with open(testset_metadata_name, "r") as f:
            testset_metadata = json.load(f)     
        model_id = int(FLAGS.rollout_name.split('_')[1])
        print(model_id)
        print(testset_metadata[model_id])
        asp_box = []
        asp_center_location = []
        if 'asperities' in testset_metadata[model_id].keys():
            # the model contain multiple asperities
            asp_list = testset_metadata[model_id]['asperities']
            hypo = testset_metadata[model_id]['hypocenter_location_km']
            for asp in asp_list:
                asp_location = asp['asperity_location_km']
                hw = asp['asperity_half_square_size_km']
                asp_center_location.append(asp_location)
                asp_box.append([[asp_location[0]+hw,asp_location[1]+hw], 
                           [asp_location[0]+hw,asp_location[1]-hw],
                           [asp_location[0]-hw,asp_location[1]-hw],
                           [asp_location[0]-hw,asp_location[1]+hw], 
                           [asp_location[0]+hw,asp_location[1]+hw]])   
                     
        else:
            # the model contain only one asperity
            asp = testset_metadata[model_id]['asperity_location_km']
            hw = testset_metadata[model_id]['asperity_half_square_size_km']
            hypo = testset_metadata[model_id]['hypocenter_location_km']
            asp_center_location.append(asp)
            asp_box.append([[asp[0]+hw,asp[1]+hw], 
                       [asp[0]+hw,asp[1]-hw],
                       [asp[0]-hw,asp[1]-hw],
                       [asp[0]-hw,asp[1]+hw], 
                       [asp[0]+hw,asp[1]+hw]])        
    
    # read rollout data
    with open(rollout_path, 'rb') as f:
        result = pickle.load(f)
    ground_truth_vel = np.concatenate((result["initial_velocities"], result["ground_truth_rollout"]))
    predicted_vel = np.concatenate((result["initial_velocities"], result["predicted_rollout"]))

    # compute velocity magnitude
    ground_truth_vel_magnitude = np.linalg.norm(ground_truth_vel, axis=-1)
    predicted_vel_magnitude = np.linalg.norm(predicted_vel, axis=-1)
    velocity_result = {
        "ground_truth": ground_truth_vel_magnitude,
        "prediction": predicted_vel_magnitude,
        "Difference": ground_truth_vel_magnitude - predicted_vel_magnitude
    }

    # variables for render
    n_timesteps = len(ground_truth_vel_magnitude)
    triang = tri.Triangulation(result["node_coords"][0][:, 0], result["node_coords"][0][:, 1])
    x_coor = result["node_coords"][0][:, 0]
    z_coor = result["node_coords"][0][:, 1]
    
    def find_node_index(x0, z0, x_coor, z_coor):
        x = np.abs(x_coor - x0)
        z = np.abs(z_coor - z0)
        index = np.argmin(x + z)
        return index
    
    
    # color
    ti = np.int32(5)
    vmin = np.concatenate(
        (result["predicted_rollout"][ti][:, 0], result["ground_truth_rollout"][ti][:, 0])).min()
    vmax = np.concatenate(
        (result["predicted_rollout"][ti][:, 0], result["ground_truth_rollout"][ti][:, 0])).max()

    #vmin = 0.0
    #vmax = 10.
    fig = plt.figure(figsize=(8, 5))
    def animate(i):
        fig.clear()
        grid = ImageGrid(fig, 111,
                        nrows_ncols=(1, 3),
                         axes_pad=0.3,
                         share_all=True,
                         cbar_location="right",
                         cbar_mode="single",
                         cbar_size="1.5%",
                         cbar_pad=0.15)

        for j, (sim, vel) in enumerate(velocity_result.items()):

            #grid[j].triplot(triang, 'o-', color='k', ms=0.5, lw=0.3)
            handle = grid[j].tripcolor(triang, vel[i], cmap="RdBu_r", vmax=vmax, vmin=vmin)
            fig.colorbar(handle, cax=grid.cbar_axes[0])
            grid[j].set_title(sim)

            if metadata_file_exists==True:
                for iasp in range(len(asp_box)):
                    asp_box_ = np.array(asp_box[iasp])
                    grid[j].plot(asp_box_[:,0]*1e3, asp_box_[:,1]*1e3, 'r')
                grid[j].plot(hypo[0]*1e3, hypo[1]*1e3, 'r*', markersize=20)
        
            if j==2: 
                grid[j].set_xlabel('Lat grid')
            grid[j].set_ylabel('Lon grid')

    # Creat animation
    ani = animation.FuncAnimation(
        fig, animate, frames=np.arange(0, n_timesteps, FLAGS.step_stride), interval=100)

    ani.save(f'{animation_filename}', dpi=50, fps=10, writer='imagemagick')
    print(f"Animation saved to: {animation_filename}")


def main(_):
    render_gif_animation()


if __name__ == '__main__':
    app.run(main)
