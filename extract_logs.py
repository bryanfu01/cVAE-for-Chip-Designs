import os
import math
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def extract_and_plot_tb_logs(log_dir, save_path="/content/drive/MyDrive/ECE_175B_Final_Project/soft_drc_loss_curves.png"):
    print(f"Loading TensorBoard logs from: {log_dir}")
    
    event_acc = EventAccumulator(log_dir, size_guidance={'scalars': 0})
    event_acc.Reload()

    available_tags = event_acc.Tags()['scalars']
    print(f"Found metrics: {available_tags}")
    
    # Build Step-to-Epoch mapping
    step_to_epoch = {}
    if 'epoch' in available_tags:
        for e in event_acc.Scalars('epoch'):
            step_to_epoch[e.step] = e.value
            
    def get_epoch(step):
        if not step_to_epoch: return step
        if step in step_to_epoch: return step_to_epoch[step]
        past_steps = [s for s in step_to_epoch.keys() if s <= step]
        return step_to_epoch[max(past_steps)] if past_steps else 0

    # Define the two categories of metrics to track
    base_metrics = ['loss', 'val_loss', 'Reconstruction_Loss', 'KLD']
    
    # UPDATE THESE names to match exactly what you pass to self.log() in experiment.py
    physics_metrics = ['Area_Loss', 'Overlap_Loss', 'Thermal_Loss', 'Sharpness_Loss', 'Cohesion_Loss', 'Scaled_DRC_Loss']
    
    history = {}
    for tag in base_metrics + physics_metrics:
        # Sometimes Lightning prepends 'train/' or 'val/' to custom logs
        actual_tags = [t for t in available_tags if tag in t]
        for act_tag in actual_tags:
            events = event_acc.Scalars(act_tag)
            epochs = [get_epoch(e.step) for e in events]
            values = [e.value for e in events]
            
            # Flip negative KLD for standard viewing
            if 'KLD' in act_tag:
                values = [-v for v in values]
                
            history[act_tag] = {'epochs': epochs, 'values': values}

    # --- Plotting Architecture ---
    if not history:
        print("No matching metrics found to plot.")
        return

    # Calculate grid size dynamically based on how many metrics we found
    num_plots = len(history)
    cols = 3
    rows = math.ceil(num_plots / cols)
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4))
    axes = axes.flatten() # Make indexing easier
    
    for i, (metric_name, data) in enumerate(history.items()):
        ax = axes[i]
        
        # Color coding: Green for Recon, Purple for KLD, Red/Orange for Physics
        color = 'blue'
        if 'Reconstruction' in metric_name: color = 'green'
        elif 'KLD' in metric_name: color = 'purple'
        elif any(p in metric_name for p in physics_metrics): color = 'tomato'
            
        ax.plot(data['epochs'], data['values'], color=color, linewidth=2)
        
        # Formatting
        ax.set_title(metric_name.replace('train/', '').replace('val/', ''), fontsize=12, fontweight='bold')
        ax.set_xlabel("Epochs")
        ax.set_ylabel("Value")
        ax.grid(True, alpha=0.3)

    # Hide any unused subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"\nSaved epoch-based plots to: {save_path}")
    plt.show()

if __name__ == "__main__":
    # CHANGE THIS to the folder containing your vanilla run's events.out.tfevents file
    SOFT_DRC_LOG_DIR = "/content/checkpoints/ConditionalVAE/version_0/events.out.tfevents.1780351797.05ddb3851851.26722.0"
    
    extract_and_plot_tb_logs(SOFT_DRC_LOG_DIR)