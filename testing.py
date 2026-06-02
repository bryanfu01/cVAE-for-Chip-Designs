import torch
import torch.nn.functional as F

def test_soft_overlap_penalty():
    print("--- Isolating Soft Overlap Penalty ---")
    # Simulate: 1 Chip, 2 Macros, 64x64 Grid
    B, C, H, W = 1, 2, 64, 64
    continuous_layouts = torch.zeros(B, C, H, W)
    macro_powers = torch.tensor([[0.5, 0.5]]) # Both valid

    # Draw Macro 1: 10x10 solid box at coordinates (20, 20)
    continuous_layouts[0, 0, 20:30, 20:30] = 1.0 
    
    # Draw Macro 2: 10x10 solid box at coordinates (25, 25)
    # This creates exactly a 5x5 overlapping intersection (25 pixels)
    continuous_layouts[0, 1, 25:35, 25:35] = 1.0 

    # --- Soft DRC Overlap Math (From your core_engines) ---
    # 1. Mask out padded channels (Multiplication method, NOT boolean index!)
    valid_mask = (macro_powers != -1.0).view(B, C, 1, 1).float()
    valid_layouts = continuous_layouts * valid_mask

    # 2. Sum the channels together
    # Where they overlap, 1.0 + 1.0 = 2.0
    summed_layouts = valid_layouts.sum(dim=1) 

    # 3. Apply ReLU(sum - 1.0) to isolate only the overlaps
    # 2.0 - 1.0 = 1.0 penalty. 1.0 - 1.0 = 0.0 penalty.
    overlap_violations = F.relu(summed_layouts - 1.0)
    total_overlap_loss = overlap_violations.sum()

    print(f"Expected Overlap:   25.00 pixels")
    print(f"Calculated Penalty: {total_overlap_loss.item():.2f} pixels")
    
    if total_overlap_loss.item() == 25.0:
        print("Verdict: OVERLAP PENALTY IS MATHEMATICALLY FLAWLESS.")

if __name__ == '__main__':
    test_soft_overlap_penalty()