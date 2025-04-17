import pickle
from absl import app
from absl import flags
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.tri as tri
from mpl_toolkits.axes_grid1 import ImageGrid
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib import animation
from matplotlib.animation import FFMpegWriter
import matplotlib as mpl
import json, os

# pip install imageio[ffmepg] # to add mp4 support

flags.DEFINE_string("rollout_dir", None, help="Directory where rollout.pkl are located")
flags.DEFINE_string("rollout_name", None, help="Name of rollout `.pkl` file")
flags.DEFINE_integer("step_stride", 3, help="Stride of steps to skip.")
FLAGS = flags.FLAGS

def render_gif_animation():

    rollout_path = f"{FLAGS.rollout_dir}/{FLAGS.rollout_name}.pkl"
    movie_format = 'mp4' # gif or 'mp4'
    animation_filename = f"{FLAGS.rollout_dir}/{FLAGS.rollout_name}.{movie_format}" #.gif"
    
    # if testset metadata exists, use the info for better rendering.
    testset_metadata_name = f"{FLAGS.rollout_dir}/testset_metadata.json"
    metadata_file_exists = os.path.exists(testset_metadata_name)
    is_asperity = False

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
            is_asperity = True
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
                     
        elif 'asperity_location_km' in testset_metadata[model_id].keys():
            # the model contain only one asperity
            is_asperity = True
            asp = testset_metadata[model_id]['asperity_location_km']
            hw = testset_metadata[model_id]['asperity_half_square_size_km']
            hypo = testset_metadata[model_id]['hypocenter_location_km']
            asp_center_location.append(asp)
            asp_box.append([[asp[0]+hw,asp[1]+hw], 
                       [asp[0]+hw,asp[1]-hw],
                       [asp[0]-hw,asp[1]-hw],
                       [asp[0]-hw,asp[1]+hw], 
                       [asp[0]+hw,asp[1]+hw]])        
        else:
           hypo = testset_metadata[model_id]['hypocenter_location_km']
           print('No apserities.')
    
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
        "Abs difference": np.abs(ground_truth_vel_magnitude - predicted_vel_magnitude)
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
    
    # pick individual on-fault station for time series plotting
    if is_asperity == True:
        hpyo_index = find_node_index(hypo[0], hypo[1], x_coor, z_coor)
        asp_index = []
        num_of_asperities = len(asp_center_location)
        if num_of_asperities >1:
            for i, asp in enumerate(asp_center_location):
                asp_index.append(find_node_index(asp[0], asp[1], x_coor, z_coor))

            single_st_label = ['Hypo', 'ASP1', 'ASP2']
        else:
            asp_index.append(find_node_index(asp_center_location[0][0], asp_center_location[0][1], x_coor, z_coor))
            asp_index.append(find_node_index(0, -5, x_coor, z_coor))
            single_st_label = ['Hypo', 'ASP1', '0.-5']
    else: # pick three fixed stations
        hpyo_index = find_node_index(0.0, -5, x_coor, z_coor)
        asp_index = []
        asp_index.append(find_node_index(5, -5, x_coor, z_coor))
        asp_index.append(find_node_index(-5, -5, x_coor, z_coor))
        single_st_label = ['0.-5', '5.-5', '-5.-5']

    # color
    ti = np.int32(n_timesteps/3)
    vmin = np.concatenate(
        (result["predicted_rollout"][ti][:, 0], result["ground_truth_rollout"][ti][:, 0])).min()
    vmax = np.concatenate(
        (result["predicted_rollout"][ti][:, 0], result["ground_truth_rollout"][ti][:, 0])).max()

    #vmin = 0.0
    #vmax = 10.
    time_steps = []
    hypo_ts = {sim: [] for sim in velocity_result.keys()}
    asp_ts = []
    for iasp in range(len(asp_index)):
        asp_ts.append({sim: [] for sim in velocity_result.keys()})

    # Init figures
    fig = plt.figure(figsize=(5, 8))
    # Define grid with 4 rows: 3 for ImageGrid, 1 for time series (adjust height ratios)
    gs = GridSpec(4, 1, height_ratios=[1, 1, 1, 0.4], hspace=0.3)

    def animate(i):
        #print(f"Render step {i}/{n_timesteps}")

        fig.clear()
        grid = ImageGrid(fig, gs[0:3],
                         nrows_ncols=(3, 1),
                         axes_pad=0.3,
                         share_all=True,
                         cbar_location="right",
                         cbar_mode="single",
                         cbar_size="1.5%",
                         cbar_pad=0.15)
        inner_gs = GridSpecFromSubplotSpec(1, 3, subplot_spec=gs[3], hspace=0.1)
        ts_axes = [fig.add_subplot(inner_gs[j]) for j in range(3)]
        time_steps.append(i)

        for j, (sim, vel) in enumerate(velocity_result.items()):

            #grid[j].triplot(triang, 'o-', color='k', ms=0.5, lw=0.3)
            #grid[j].triplot(triang, color='gray', lw=0.1)
            handle = grid[j].tripcolor(triang, vel[i], vmax=vmax, vmin=vmin)
            fig.colorbar(handle, cax=grid.cbar_axes[0])
            grid[j].set_title(sim)

            if metadata_file_exists==True:
                for iasp in range(len(asp_box)):
                    asp_box_ = np.array(asp_box[iasp])
                    grid[j].plot(asp_box_[:,0]*1e3, asp_box_[:,1]*1e3, 'r')
                grid[j].plot(hypo[0]*1e3, hypo[1]*1e3, 'r*', markersize=20)
        
            if j==2: 
                grid[j].set_xlabel('Strike, m')
            grid[j].set_ylabel('Dip, m')

            #if j<2:
            hypo_ts[sim].append(vel[i][hpyo_index])
            for k in range(len(asp_index)):
                asp_ts[k][sim].append(vel[i][asp_index[k]])

        # plot time series
        def plot_st(ts_ax, time_steps, vel_hist, st_name):
            for j, sim in enumerate(vel_hist.keys()):
                if j < 2:
                    marker = 'k-' if j==0 else 'r:'
                    ts_ax.plot(time_steps, vel_hist[sim], marker, lw=0.5, label=st_name)
                    #ts_ax.set_xlabel('Time step')
                    #ts_ax.set_ylabel('Sliprate, m/s')
                    ts_ax.legend(loc='upper right', fontsize=6)
                    ts_ax.set_xlim(0, n_timesteps)
                    ts_ax.set_ylim(0, vmax)
        
        plot_st(ts_axes[0], time_steps, hypo_ts, single_st_label[0])
        ts_axes[0].set_xlabel(f"k-GT; r-pred. {single_st_label[0]}")
        ts_axes[0].set_ylabel("Sliprate, m/s")
        plot_st(ts_axes[1], time_steps, asp_ts[0], single_st_label[1])
        ts_axes[1].set_xlabel(f"Time step. {single_st_label[1]}")
        plot_st(ts_axes[2], time_steps, asp_ts[1], single_st_label[2])
        ts_axes[2].set_xlabel(f"{single_st_label[2]}")

    # Creat animation

    ani = animation.FuncAnimation(
        fig, animate, frames=np.arange(0, n_timesteps, FLAGS.step_stride), interval=20)

    if movie_format == "gif":
        ani.save(f'{animation_filename}', dpi=100, fps=30, writer='imagemagick')
    elif movie_format == "mp4":
        import imageio_ffmpeg
        ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
        mpl.rcParams['animation.ffmpeg_path'] = ffmpeg_path
        writer = FFMpegWriter(fps=30, bitrate=1800)
        ani.save(f'{animation_filename}', dpi=100, writer=writer)

    print(f"Animation saved to: {animation_filename}")


def main(_):
    render_gif_animation()


if __name__ == '__main__':
    app.run(main)
