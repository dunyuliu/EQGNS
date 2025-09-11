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

# Updated model list to include new nmp10.cotopaxi.r1 path
model_list = [#"../results/case3.200m.homo.a.Vw/nmp10.cotopaxi/", 
              #"../results/case4.200m.multi.stress.homo.a.Vw/nmp10.cotopaxi/",
              #"../results/case4.200m.multi.stress.homo.a.Vw/nmp10.cotopaxi.r1/",
              #"../results/case4.200m.multi.stress.homo.a.Vw/nmp10.nnode4.cotopaxi.r1/",
              #"../results/case4.200m.multi.stress.160scenarios.homo.a.Vw/nmp10.cotopaxi.r1/",
              #"../results/case4.200m.multi.stress.160scenarios.homo.a.Vw/nmp10.b4.cotopaxi.r1/",
              "../results/case4.200m.multi.stress.160scenarios.homo.a.Vw/nmp10.lr3e-5.b8.cotopaxi.r1/"]

for model_path in model_list:
    loss_log = np.loadtxt(model_path+'loss_log.txt')
    
    # Check if this is the new format (7 columns) or old format (3 columns)
    if loss_log.shape[1] == 7:
        # New format: step, loss, valid_loss, current_vel_rms, target_vel_rms, current_vel_max, target_vel_max
        is_new_format = True
    else:
        # Old format: step, loss, valid_loss
        is_new_format = False
    
    loss_log_coarse = []
    itag = 0
    for i in range(loss_log.shape[0]):
        if i%nstep == 0:
            loss_log_coarse.append(loss_log[i,:])
            itag += 1

    loss_log_coarse = np.array(loss_log_coarse)

    # Plot loss curves
    fig, ax = plt.subplots(figsize=(5,4))
    ax.plot(loss_log_coarse[:,0], loss_log_coarse[:,1], label='train')
    ax.plot(loss_log_coarse[:,0], loss_log_coarse[:,2], label='valid')

    ax.set_yscale('log')
    ax.set_xlabel("Training steps")
    ax.set_ylabel(f"Loss evaluated every {1000*nstep} steps")
    fig.tight_layout()
    ax.legend()

    fig.savefig(f"{model_path}train_valid_loss_per_{1000*nstep}_steps.png", dpi=300)
    
    # If new format, create additional plots for velocity metrics and correlations
    if is_new_format:
        # Plot velocity RMS comparison
        fig2, ax2 = plt.subplots(figsize=(5,4))
        ax2.plot(loss_log_coarse[:,0], loss_log_coarse[:,3], label='current vel RMS')
        ax2.plot(loss_log_coarse[:,0], loss_log_coarse[:,4], label='target vel RMS')
        
        ax2.set_xlabel("Training steps")
        ax2.set_ylabel("Velocity RMS")
        ax2.set_yscale('log')
        fig2.tight_layout()
        ax2.legend()
        
        fig2.savefig(f"{model_path}vel_rms_per_{1000*nstep}_steps.png", dpi=300)
        
        # Plot velocity max comparison
        fig3, ax3 = plt.subplots(figsize=(5,4))
        ax3.plot(loss_log_coarse[:,0], loss_log_coarse[:,5], label='current vel max')
        ax3.plot(loss_log_coarse[:,0], loss_log_coarse[:,6], label='target vel max')
        
        ax3.set_xlabel("Training steps")
        ax3.set_ylabel("Velocity Max")
        ax3.set_yscale('log')
        fig3.tight_layout()
        ax3.legend()
        
        fig3.savefig(f"{model_path}vel_max_per_{1000*nstep}_steps.png", dpi=300)
        
        # Plot loss normalized by current velocity RMS (dense)
        fig5, ax5 = plt.subplots(figsize=(5,4))
        # Avoid division by zero by adding small epsilon
        vel_rms_safe = np.maximum(loss_log_coarse[:,3], 1e-10)
        train_loss_norm = loss_log_coarse[:,1] / vel_rms_safe
        valid_loss_norm = loss_log_coarse[:,2] / vel_rms_safe
        
        ax5.plot(loss_log_coarse[:,0], valid_loss_norm, label='valid loss / vel RMS', color='orange')
        ax5.plot(loss_log_coarse[:,0], train_loss_norm, label='train loss / vel RMS', color='blue')
        
        ax5.set_yscale('log')
        ax5.set_ylim(1e-5, 1e0)
        ax5.set_xlabel("Training steps")
        ax5.set_ylabel("Loss normalized by Vel RMS")
        ax5.set_title("Dense sampling")
        fig5.tight_layout()
        ax5.legend()
        
        fig5.savefig(f"{model_path}loss_normalized_by_vel_rms_dense.png", dpi=300)
        
        # Plot loss normalized by current velocity RMS with coarser sampling
        nstep_coarse = 50  # Much coarser sampling
        loss_log_extra_coarse = []
        for i in range(0, loss_log.shape[0], nstep_coarse):
            loss_log_extra_coarse.append(loss_log[i,:])
        loss_log_extra_coarse = np.array(loss_log_extra_coarse)
        
        fig6, ax6 = plt.subplots(figsize=(5,4))
        vel_rms_safe_coarse = np.maximum(loss_log_extra_coarse[:,3], 1e-10)
        train_loss_norm_coarse = loss_log_extra_coarse[:,1] / vel_rms_safe_coarse
        valid_loss_norm_coarse = loss_log_extra_coarse[:,2] / vel_rms_safe_coarse
        
        ax6.plot(loss_log_extra_coarse[:,0], valid_loss_norm_coarse, label='valid loss / vel RMS', color='orange')
        ax6.plot(loss_log_extra_coarse[:,0], train_loss_norm_coarse, label='train loss / vel RMS', color='blue')
        
        ax6.set_yscale('log')
        ax6.set_ylim(1e-5, 1e0)
        ax6.set_xlabel("Training steps")
        ax6.set_ylabel("Loss normalized by Vel RMS")
        ax6.set_title(f"Coarse sampling (every {nstep_coarse*1000} steps)")
        fig6.tight_layout()
        ax6.legend()
        
        fig6.savefig(f"{model_path}loss_normalized_by_vel_rms_coarse.png", dpi=300)
        
        # Plot loss normalized by target velocity RMS (dense)
        fig7, ax7 = plt.subplots(figsize=(5,4))
        vel_rms_target_safe = np.maximum(loss_log_coarse[:,4], 1e-10)
        train_loss_norm_target = loss_log_coarse[:,1] / vel_rms_target_safe
        valid_loss_norm_target = loss_log_coarse[:,2] / vel_rms_target_safe
        
        ax7.plot(loss_log_coarse[:,0], valid_loss_norm_target, label='valid loss / target vel RMS', color='orange')
        ax7.plot(loss_log_coarse[:,0], train_loss_norm_target, label='train loss / target vel RMS', color='blue')
        
        ax7.set_yscale('log')
        ax7.set_ylim(1e-5, 1e0)
        ax7.set_xlabel("Training steps")
        ax7.set_ylabel("Loss normalized by Target Vel RMS")
        ax7.set_title("Dense sampling")
        fig7.tight_layout()
        ax7.legend()
        
        fig7.savefig(f"{model_path}loss_normalized_by_target_vel_rms_dense.png", dpi=300)
        
        # Plot loss normalized by target velocity RMS (coarse)
        fig7b, ax7b = plt.subplots(figsize=(5,4))
        vel_rms_target_safe_coarse = np.maximum(loss_log_extra_coarse[:,4], 1e-10)
        train_loss_norm_target_coarse = loss_log_extra_coarse[:,1] / vel_rms_target_safe_coarse
        valid_loss_norm_target_coarse = loss_log_extra_coarse[:,2] / vel_rms_target_safe_coarse
        
        ax7b.plot(loss_log_extra_coarse[:,0], valid_loss_norm_target_coarse, label='valid loss / target vel RMS', color='orange')
        ax7b.plot(loss_log_extra_coarse[:,0], train_loss_norm_target_coarse, label='train loss / target vel RMS', color='blue')
        
        ax7b.set_yscale('log')
        ax7b.set_ylim(1e-5, 1e0)
        ax7b.set_xlabel("Training steps")
        ax7b.set_ylabel("Loss normalized by Target Vel RMS")
        ax7b.set_title(f"Coarse sampling (every {nstep_coarse*1000} steps)")
        fig7b.tight_layout()
        ax7b.legend()
        
        fig7b.savefig(f"{model_path}loss_normalized_by_target_vel_rms_coarse.png", dpi=300)
        
        # Plot loss normalized by current velocity MAX (dense)
        fig8, ax8 = plt.subplots(figsize=(5,4))
        vel_max_safe = np.maximum(loss_log_coarse[:,5], 1e-10)
        train_loss_norm_max = loss_log_coarse[:,1] / vel_max_safe
        valid_loss_norm_max = loss_log_coarse[:,2] / vel_max_safe
        
        ax8.plot(loss_log_coarse[:,0], valid_loss_norm_max, label='valid loss / current vel MAX', color='orange')
        ax8.plot(loss_log_coarse[:,0], train_loss_norm_max, label='train loss / current vel MAX', color='blue')
        
        ax8.set_yscale('log')
        ax8.set_ylim(1e-5, 1e0)
        ax8.set_xlabel("Training steps")
        ax8.set_ylabel("Loss normalized by Current Vel MAX")
        ax8.set_title("Dense sampling")
        fig8.tight_layout()
        ax8.legend()
        
        fig8.savefig(f"{model_path}loss_normalized_by_vel_max_dense.png", dpi=300)
        
        # Plot loss normalized by current velocity MAX (coarse)
        fig8b, ax8b = plt.subplots(figsize=(5,4))
        vel_max_safe_coarse = np.maximum(loss_log_extra_coarse[:,5], 1e-10)
        train_loss_norm_max_coarse = loss_log_extra_coarse[:,1] / vel_max_safe_coarse
        valid_loss_norm_max_coarse = loss_log_extra_coarse[:,2] / vel_max_safe_coarse
        
        ax8b.plot(loss_log_extra_coarse[:,0], valid_loss_norm_max_coarse, label='valid loss / current vel MAX', color='orange')
        ax8b.plot(loss_log_extra_coarse[:,0], train_loss_norm_max_coarse, label='train loss / current vel MAX', color='blue')
        
        ax8b.set_yscale('log')
        ax8b.set_ylim(1e-5, 1e0)
        ax8b.set_xlabel("Training steps")
        ax8b.set_ylabel("Loss normalized by Current Vel MAX")
        ax8b.set_title(f"Coarse sampling (every {nstep_coarse*1000} steps)")
        fig8b.tight_layout()
        ax8b.legend()
        
        fig8b.savefig(f"{model_path}loss_normalized_by_vel_max_coarse.png", dpi=300)
        
        # Plot loss normalized by target velocity MAX (dense)
        fig9, ax9 = plt.subplots(figsize=(5,4))
        vel_max_target_safe = np.maximum(loss_log_coarse[:,6], 1e-10)
        train_loss_norm_max_target = loss_log_coarse[:,1] / vel_max_target_safe
        valid_loss_norm_max_target = loss_log_coarse[:,2] / vel_max_target_safe
        
        ax9.plot(loss_log_coarse[:,0], valid_loss_norm_max_target, label='valid loss / target vel MAX', color='orange')
        ax9.plot(loss_log_coarse[:,0], train_loss_norm_max_target, label='train loss / target vel MAX', color='blue')
        
        ax9.set_yscale('log')
        ax9.set_ylim(1e-5, 1e0)
        ax9.set_xlabel("Training steps")
        ax9.set_ylabel("Loss normalized by Target Vel MAX")
        ax9.set_title("Dense sampling")
        fig9.tight_layout()
        ax9.legend()
        
        fig9.savefig(f"{model_path}loss_normalized_by_target_vel_max_dense.png", dpi=300)
        
        # Plot loss normalized by target velocity MAX (coarse)
        fig9b, ax9b = plt.subplots(figsize=(5,4))
        vel_max_target_safe_coarse = np.maximum(loss_log_extra_coarse[:,6], 1e-10)
        train_loss_norm_max_target_coarse = loss_log_extra_coarse[:,1] / vel_max_target_safe_coarse
        valid_loss_norm_max_target_coarse = loss_log_extra_coarse[:,2] / vel_max_target_safe_coarse
        
        ax9b.plot(loss_log_extra_coarse[:,0], valid_loss_norm_max_target_coarse, label='valid loss / target vel MAX', color='orange')
        ax9b.plot(loss_log_extra_coarse[:,0], train_loss_norm_max_target_coarse, label='train loss / target vel MAX', color='blue')
        
        ax9b.set_yscale('log')
        ax9b.set_ylim(1e-5, 1e0)
        ax9b.set_xlabel("Training steps")
        ax9b.set_ylabel("Loss normalized by Target Vel MAX")
        ax9b.set_title(f"Coarse sampling (every {nstep_coarse*1000} steps)")
        fig9b.tight_layout()
        ax9b.legend()
        
        fig9b.savefig(f"{model_path}loss_normalized_by_target_vel_max_coarse.png", dpi=300)
        
        # Plot original loss (dense sampling) 
        fig10, ax10 = plt.subplots(figsize=(5,4))
        ax10.plot(loss_log_coarse[:,0], loss_log_coarse[:,2], label='valid loss', color='orange')
        ax10.plot(loss_log_coarse[:,0], loss_log_coarse[:,1], label='train loss', color='blue')
        
        ax10.set_yscale('log')
        ax10.set_ylim(1e-5, 1e0)
        ax10.set_xlabel("Training steps")
        ax10.set_ylabel("Original Loss")
        ax10.set_title("Original Loss (dense sampling)")
        fig10.tight_layout()
        ax10.legend()
        
        fig10.savefig(f"{model_path}original_loss_dense.png", dpi=300)
        
        # Plot original loss (coarse sampling)
        fig11, ax11 = plt.subplots(figsize=(5,4))
        ax11.plot(loss_log_extra_coarse[:,0], loss_log_extra_coarse[:,2], label='valid loss', color='orange')
        ax11.plot(loss_log_extra_coarse[:,0], loss_log_extra_coarse[:,1], label='train loss', color='blue')
        
        ax11.set_yscale('log')
        ax11.set_ylim(1e-5, 1e0)
        ax11.set_xlabel("Training steps")
        ax11.set_ylabel("Original Loss")
        ax11.set_title(f"Original Loss (coarse sampling - every {nstep_coarse*1000} steps)")
        fig11.tight_layout()
        ax11.legend()
        
        fig11.savefig(f"{model_path}original_loss_coarse.png", dpi=300)
        
        # Plot current vs target velocity correlation
        fig12, ((ax12a, ax12b), (ax12c, ax12d)) = plt.subplots(2, 2, figsize=(10,8))
        
        # Current vs Target Vel RMS
        ax12a.scatter(loss_log_coarse[:,4], loss_log_coarse[:,3], alpha=0.6, s=20)
        ax12a.set_xlabel("Target Vel RMS")
        ax12a.set_ylabel("Current Vel RMS")
        ax12a.set_title("Current vs Target Vel RMS")
        ax12a.set_xscale('log')
        ax12a.set_yscale('log')
        
        # Current vs Target Vel MAX
        ax12b.scatter(loss_log_coarse[:,6], loss_log_coarse[:,5], alpha=0.6, s=20, color='orange')
        ax12b.set_xlabel("Target Vel MAX")
        ax12b.set_ylabel("Current Vel MAX")
        ax12b.set_title("Current vs Target Vel MAX")
        ax12b.set_xscale('log')
        ax12b.set_yscale('log')
        
        # RMS vs MAX (Current)
        ax12c.scatter(loss_log_coarse[:,5], loss_log_coarse[:,3], alpha=0.6, s=20, color='green')
        ax12c.set_xlabel("Current Vel MAX")
        ax12c.set_ylabel("Current Vel RMS")
        ax12c.set_title("Current: RMS vs MAX")
        ax12c.set_xscale('log')
        ax12c.set_yscale('log')
        
        # RMS vs MAX (Target)
        ax12d.scatter(loss_log_coarse[:,6], loss_log_coarse[:,4], alpha=0.6, s=20, color='red')
        ax12d.set_xlabel("Target Vel MAX")
        ax12d.set_ylabel("Target Vel RMS")
        ax12d.set_title("Target: RMS vs MAX")
        ax12d.set_xscale('log')
        ax12d.set_yscale('log')
        
        fig12.tight_layout()
        fig12.savefig(f"{model_path}velocity_correlations.png", dpi=300)
        
        # Calculate velocity correlations
        current_target_rms_corr = np.corrcoef(loss_log_coarse[:,3], loss_log_coarse[:,4])[0,1]
        current_target_max_corr = np.corrcoef(loss_log_coarse[:,5], loss_log_coarse[:,6])[0,1]
        current_rms_max_corr = np.corrcoef(loss_log_coarse[:,3], loss_log_coarse[:,5])[0,1]
        target_rms_max_corr = np.corrcoef(loss_log_coarse[:,4], loss_log_coarse[:,6])[0,1]
        
        print(f"\nVelocity correlations:")
        print(f"Current vs Target Vel RMS: {current_target_rms_corr:.3f}")
        print(f"Current vs Target Vel MAX: {current_target_max_corr:.3f}")
        print(f"Current RMS vs MAX: {current_rms_max_corr:.3f}")
        print(f"Target RMS vs MAX: {target_rms_max_corr:.3f}")
        
        # Correlation analysis plots
        fig4, ((ax4a, ax4b), (ax4c, ax4d)) = plt.subplots(2, 2, figsize=(10,8))
        
        # Train loss vs velocity RMS
        ax4a.scatter(loss_log_coarse[:,3], loss_log_coarse[:,1], alpha=0.6, s=20)
        ax4a.set_xlabel("Current Vel RMS")
        ax4a.set_ylabel("Train Loss")
        ax4a.set_yscale('log')
        ax4a.set_title("Train Loss vs Vel RMS")
        
        # Valid loss vs velocity RMS  
        ax4b.scatter(loss_log_coarse[:,3], loss_log_coarse[:,2], alpha=0.6, s=20, color='orange')
        ax4b.set_xlabel("Current Vel RMS")  
        ax4b.set_ylabel("Valid Loss")
        ax4b.set_yscale('log')
        ax4b.set_title("Valid Loss vs Vel RMS")
        
        # Train loss vs velocity Max
        ax4c.scatter(loss_log_coarse[:,5], loss_log_coarse[:,1], alpha=0.6, s=20)
        ax4c.set_xlabel("Current Vel Max")
        ax4c.set_ylabel("Train Loss") 
        ax4c.set_yscale('log')
        ax4c.set_title("Train Loss vs Vel Max")
        
        # Valid loss vs velocity Max
        ax4d.scatter(loss_log_coarse[:,5], loss_log_coarse[:,2], alpha=0.6, s=20, color='orange')
        ax4d.set_xlabel("Current Vel Max")
        ax4d.set_ylabel("Valid Loss")
        ax4d.set_yscale('log') 
        ax4d.set_title("Valid Loss vs Vel Max")
        
        fig4.tight_layout()
        fig4.savefig(f"{model_path}loss_velocity_correlations.png", dpi=300)
        
        # Calculate correlation coefficients for characterization
        import numpy as np
        train_vel_rms_corr = np.corrcoef(loss_log_coarse[:,1], loss_log_coarse[:,3])[0,1]
        valid_vel_rms_corr = np.corrcoef(loss_log_coarse[:,2], loss_log_coarse[:,3])[0,1]
        train_vel_max_corr = np.corrcoef(loss_log_coarse[:,1], loss_log_coarse[:,5])[0,1]
        valid_vel_max_corr = np.corrcoef(loss_log_coarse[:,2], loss_log_coarse[:,5])[0,1]
        
        # Print correlation summary
        print(f"\nCorrelation analysis for {model_path}:")
        print(f"Train Loss vs Vel RMS: {train_vel_rms_corr:.3f}")
        print(f"Valid Loss vs Vel RMS: {valid_vel_rms_corr:.3f}")
        print(f"Train Loss vs Vel Max: {train_vel_max_corr:.3f}")
        print(f"Valid Loss vs Vel Max: {valid_vel_max_corr:.3f}")
        
        # Loss range analysis
        train_loss_range = np.max(loss_log_coarse[:,1]) / np.min(loss_log_coarse[:,1])
        valid_loss_range = np.max(loss_log_coarse[:,2]) / np.min(loss_log_coarse[:,2])
        vel_rms_range = np.max(loss_log_coarse[:,3]) / np.min(loss_log_coarse[:,3])
        vel_max_range = np.max(loss_log_coarse[:,5]) / np.min(loss_log_coarse[:,5])
        
        print(f"Train Loss range (max/min): {train_loss_range:.1f}")
        print(f"Valid Loss range (max/min): {valid_loss_range:.1f}")
        print(f"Vel RMS range (max/min): {vel_rms_range:.1f}")
        print(f"Vel Max range (max/min): {vel_max_range:.1f}")

# plt.show()  # Comment out to avoid GUI display