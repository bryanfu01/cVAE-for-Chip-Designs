import torch
from torch import Tensor

def calculate_exact_overlap(boxes: Tensor) -> Tensor:
    """
    Calculates total overlapping area for a batch of discrete chip layouts.
    chip layout shape shape: (Batch, 4, Num Macros) -> 4 comprised of cx, cy, h, w
    """
    B, _, N = boxes.shape
    
    # 1. Create a boolean mask to ignore the -1 padding
    # mask shape: (B, N)
    valid_mask = boxes[:, 0, :] != -1
    
    # Create a 2D mask for valid pairs (B, N, N)
    # True only if BOTH macros in the comparison are valid
    valid_pairs = valid_mask.unsqueeze(2) & valid_mask.unsqueeze(1)
    
    # Remove self-comparisons (Macro 1 intersecting with Macro 1)
    diag_mask = ~torch.eye(N, dtype=torch.bool, device=boxes.device).unsqueeze(0)
    valid_pairs = valid_pairs & diag_mask
    
    # 2. Extract edge coordinates for all macros
    left = boxes[:, 0, :] - boxes[:, 3, :] / 2
    right = boxes[:, 0, :] + boxes[:, 3, :] / 2
    bottom = boxes[:, 1, :] - boxes[:, 2, :] / 2
    top = boxes[:, 1, :] + boxes[:, 2, :] / 2
    
    # 3. BROADCASTING: Compare every macro to every other macro
    # Expands (B, N) and (B, N) to an intersecting (B, N, N) grid
    overlap_left = torch.max(left.unsqueeze(2), left.unsqueeze(1))
    overlap_right = torch.min(right.unsqueeze(2), right.unsqueeze(1))
    overlap_bottom = torch.max(bottom.unsqueeze(2), bottom.unsqueeze(1))
    overlap_top = torch.min(top.unsqueeze(2), top.unsqueeze(1))
    
    # 4. Calculate overlapping width and height
    # clamp(min=0) ensures that if they don't overlap, the distance is 0, not negative
    w_overlap = torch.clamp(overlap_right - overlap_left, min=0)
    h_overlap = torch.clamp(overlap_top - overlap_bottom, min=0)
    
    # 5. Calculate area and apply the mask
    overlap_area = w_overlap * h_overlap
    overlap_area = overlap_area * valid_pairs
    
    # 6. Sum the total area per chip. 
    # Divide by 2 because broadcasting counts A->B and B->A as two separate overlaps!
    total_overlap_per_chip = overlap_area.sum(dim=(1, 2)) / 2
    
    return total_overlap_per_chip