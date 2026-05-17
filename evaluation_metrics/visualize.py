import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch

def plot_comparison(original_layout: torch.Tensor, 
                    target_heatmap: torch.Tensor, 
                    generated_layout: torch.Tensor, 
                    grid_size: int = 64,
                    save_path: str = "comparison.png"):
    """
    Generates a side-by-side comparison of the Target Heatmap, 
    the Original Layout, and the VAE's Generated Layout.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 1. Plot Target Heat Map
    # Squeeze out the channel dimension for plotting
    im = axes[0].imshow(target_heatmap.squeeze().cpu().numpy(), cmap='inferno', origin='lower')
    axes[0].set_title("Target Heat Map Condition")
    fig.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04)

    # Helper function to draw macros
    def draw_macros(ax, layout_tensor, title):
        ax.set_xlim(0, grid_size)
        ax.set_ylim(0, grid_size)
        ax.set_title(title)
        ax.set_aspect('equal')
        ax.grid(True, linestyle=':', alpha=0.6)
        
        num_macros = layout_tensor.shape[1]
        for i in range(num_macros):
            cx, cy, h, w = layout_tensor[:, i].cpu().numpy()
            if cx == -1: # Skip padding
                continue
            
            # Matplotlib patches anchor at the bottom-left corner
            bottom_left_x = cx - w/2
            bottom_left_y = cy - h/2
            
            rect = patches.Rectangle((bottom_left_x, bottom_left_y), w, h, 
                                     linewidth=1.5, edgecolor='blue', facecolor='cyan', alpha=0.5)
            ax.add_patch(rect)

    # 2. Plot Original Layout
    draw_macros(axes[1], original_layout, "Original Ground Truth Layout")

    # 3. Plot Generated Layout
    draw_macros(axes[2], generated_layout, "Generated Legalized Layout")

    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Visualization saved to {save_path}")
    plt.close()