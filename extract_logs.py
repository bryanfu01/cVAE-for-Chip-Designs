import os
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def extract_and_plot_tb_logs(log_dir, save_path="/content/drive/MyDrive/ECE_175B_Final_Project/vanilla_loss_curves.png"):
    print(f"Loading TensorBoard logs from: {log_dir}")
    
    # Initialize the event accumulator
    event_acc = EventAccumulator(log_dir, size_guidance={'scalars': 0})
    event_acc.Reload()

    available_tags = event_acc.Tags()['scalars']
    print(f"Found metrics: {available_tags}")
    
    # 1. Build a Step-to-Epoch mapping
    step_to_epoch = {}
    if 'epoch' in available_tags:
        for e in event_acc.Scalars('epoch'):
            step_to_epoch[e.step] = e.value
    else:
        print("Warning: 'epoch' tag not found. PyTorch Lightning may not have logged it.")
        
    # Helper to forward-fill epochs for steps that didn't log an exact epoch scalar
    def get_epoch(step):
        if not step_to_epoch: return step
        if step in step_to_epoch: return step_to_epoch[step]
        # Find the most recent epoch prior to this step
        past_steps = [s for s in step_to_epoch.keys() if s <= step]
        return step_to_epoch[max(past_steps)] if past_steps else 0

    target_metrics = ['Reconstruction_Loss', 'KLD', 'loss', 'val_loss']
    history = {}
    
    for tag in target_metrics:
        if tag in available_tags:
            events = event_acc.Scalars(tag)
            # Apply our mapping to shift the x-axis to Epochs
            epochs = [get_epoch(e.step) for e in events]
            values = [e.value for e in events]
            history[tag] = {'epochs': epochs, 'values': values}

    # --- Plotting the Curves ---
    plt.figure(figsize=(15, 5))
    
    # 1. Total Loss Curve
    plt.subplot(1, 3, 1)
    if 'loss' in history:
        plt.plot(history['loss']['epochs'], history['loss']['values'], label='Train Loss', alpha=0.8)
    if 'val_loss' in history:
        plt.plot(history['val_loss']['epochs'], history['val_loss']['values'], label='Val Loss', alpha=0.8)
    plt.title("Total VAE Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 2. Reconstruction Loss
    plt.subplot(1, 3, 2)
    if 'Reconstruction_Loss' in history:
        plt.plot(history['Reconstruction_Loss']['epochs'], history['Reconstruction_Loss']['values'], color='green')
    plt.title("Reconstruction Loss (BCE)")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.grid(True, alpha=0.3)

    # 3. KLD (Free Bits)
    plt.subplot(1, 3, 3)
    if 'KLD' in history:
        # Flip negative KLD to positive for standard viewing
        kld_vals = [-v for v in history['KLD']['values']]
        plt.plot(history['KLD']['epochs'], kld_vals, color='purple')
    plt.title("KL Divergence")
    plt.xlabel("Epochs")
    plt.ylabel("KLD")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"\nSaved epoch-based plots to: {save_path}")
    plt.show()

if __name__ == "__main__":
    # CHANGE THIS to the folder containing your vanilla run's events.out.tfevents file
    VANILLA_LOG_DIR = "/content/drive/MyDrive/ECE_175B_Final_Project/events.out.tfevents.1780167905.193acaa2b160.5213.0" 
    
    extract_and_plot_tb_logs(VANILLA_LOG_DIR)