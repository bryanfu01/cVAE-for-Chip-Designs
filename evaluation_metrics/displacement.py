import torch
from torch import Tensor

def calculate_displacement(pre_legalized: Tensor, post_legalized: Tensor) -> Tensor:
    """
    Calculates the average pixel displacement per macro caused by the legalizer.
    Both inputs are shape (Batch, 4, C).
    """
    # 1. Mask out the -1 padding
    valid_mask = pre_legalized[:, 0, :] != -1
    
    # 2. Extract X and Y for both states
    pre_x, pre_y = pre_legalized[:, 0, :], pre_legalized[:, 1, :]
    post_x, post_y = post_legalized[:, 0, :], post_legalized[:, 1, :]
    
    # 3. Calculate Euclidean distance (Pythagorean theorem)
    dx = pre_x - post_x
    dy = pre_y - post_y
    distances = torch.sqrt(dx**2 + dy**2)
    
    # 4. Calculate the average displacement per chip
    # Sum the valid distances and divide by the number of valid macros in that chip
    total_valid_distances = (distances * valid_mask).sum(dim=1)
    num_valid_macros = valid_mask.sum(dim=1)
    
    avg_displacement_per_chip = total_valid_distances / num_valid_macros
    
    return avg_displacement_per_chip