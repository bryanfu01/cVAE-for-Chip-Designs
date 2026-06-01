import os
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def extract_and_plot_tb_logs(log_dir, save_path="/content/drive/MyDrive/ECE_175B_Final_Project/vanilla_loss_curves.png"):
    print(f"Loading TensorBoard logs from: {log_dir}")
    
    # Initialize the event accumulator
    # size_guidance dictates how many points per scalar to keep (0 = all)
    event_acc = EventAccumulator(log_dir, size_guidance={'scalars': 0})
    event_acc.Reload()

    # List all available scalar tags in the logs
    available_tags = event_acc.Tags()['scalars']
    print(f"Found metrics: {available_tags}")
    
    # Define the core metrics we care about for the Vanilla VAE
    target_metrics = ['Reconstruction_Loss', 'KLD', 'loss', 'val_loss']
    
    # Dictionary to hold our extracted data
    history = {}
    
    for tag in target_metrics:
        if tag in available_tags:
            # Extract step and value
            events = event_acc.Scalars(tag)
            steps = [e.step for e in events]
            values = [e.value for e in events]
            history[tag] = {'steps': steps, 'values': values}
        else:
            print(f"Warning: Metric '{tag}' not found in logs.")

    # --- Plotting the Curves ---
    plt.figure(figsize=(15, 5))
    
    # 1. Total Loss Curve (Train vs Val)
    plt.subplot(1, 3, 1)
    if 'loss' in history:
        plt.plot(history['loss']['steps'], history['loss']['values'], label='Train Loss', alpha=0.8)
    if 'val_loss' in history:
        plt.plot(history['val_loss']['steps'], history['val_loss']['values'], label='Val Loss', alpha=0.8)
    plt.title("Total VAE Loss")
    plt.xlabel("Global Step")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 2. Reconstruction Loss
    plt.subplot(1, 3, 2)
    if 'Reconstruction_Loss' in history:
        plt.plot(history['Reconstruction_Loss']['steps'], history['Reconstruction_Loss']['values'], color='green')
    plt.title("Reconstruction Loss (BCE)")
    plt.xlabel("Global Step")
    plt.ylabel("Loss")
    plt.grid(True, alpha=0.3)

    # 3. KLD (Free Bits)
    plt.subplot(1, 3, 3)
    if 'KLD' in history:
        # Note: In experiment.py you logged KLD as negative, so we flip it for viewing
        kld_vals = [-v for v in history['KLD']['values']]
        plt.plot(history['KLD']['steps'], kld_vals, color='purple')
    plt.title("KL Divergence (Free Bits thresholded)")
    plt.xlabel("Global Step")
    plt.ylabel("KLD")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"\nSaved plots to: {save_path}")
    plt.show()

if __name__ == "__main__":
    # CHANGE THIS to the folder containing your vanilla run's events.out.tfevents file
    VANILLA_LOG_DIR = "/content/drive/MyDrive/ECE_175B_Final_Project/golden_weight_logs.0" 
    
    extract_and_plot_tb_logs(VANILLA_LOG_DIR)