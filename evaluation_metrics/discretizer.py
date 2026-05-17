import torch
from torch import Tensor

def extract_center_of_mass(continuous_layout: Tensor, ground_truth_macros: Tensor) -> Tensor:
    """
    Converts continuous (B, C, H, W) density back into discrete (B, 4, C) bounding boxes.
    Requires ground_truth_macros to grab the exact target Widths and Heights.
    """
    B, num_macros, H, W = continuous_layout.shape
    device = continuous_layout.device

    # 1. Create X and Y coordinate grids
    y_grid = torch.arange(H, device=device, dtype=torch.float32).view(1, 1, H, 1)
    x_grid = torch.arange(W, device=device, dtype=torch.float32).view(1, 1, 1, W)

    # 2. Calculate the Expectation (Center of Mass)
    # Add 1e-8 to prevent division by zero if a channel is completely empty
    density_sum = continuous_layout.sum(dim=(2, 3)) + 1e-8 
    
    y_c = (continuous_layout * y_grid).sum(dim=(2, 3)) / density_sum
    x_c = (continuous_layout * x_grid).sum(dim=(2, 3)) / density_sum

    # 3. Rebuild the (B, 4, C) tensor
    # Clone the ground truth to perfectly preserve the Heights, Widths, and -1 Padding!
    extracted_boxes = ground_truth_macros.clone()
    
    # 4. Inject the VAE's predicted X and Y centers only for valid (unpadded) macros
    valid_mask = ground_truth_macros[:, 0, :] != -1
    extracted_boxes[:, 0, :][valid_mask] = x_c[valid_mask]
    extracted_boxes[:, 1, :][valid_mask] = y_c[valid_mask]

    return extracted_boxes