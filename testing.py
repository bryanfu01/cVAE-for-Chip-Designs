import torch
import torch.nn.functional as F

def test_cohesion_alignment():
    # 1. Setup Mock Grid and Tensors [Batch, Channels, Height, Width]
    B, C, H, W = 1, 2, 64, 64
    layouts = torch.zeros((B, C, H, W))
    powers = torch.tensor([[10.0, 20.0]]) # Both channels are valid

    # 2. Draw perfectly isolated, solid binary macros
    # Macro 0: 10x10 block (Area = 100, Expected Perimeter = 40)
    layouts[0, 0, 5:15, 5:15] = 1.0 
    
    # Macro 1: 5x20 block (Area = 100, Expected Perimeter = 50)
    layouts[0, 1, 20:25, 10:30] = 1.0 
    
    # --- 3. Run SoftDRC TV Calculation (From soft_drcs.py) ---
    valid_mask = (powers != -1.0)
    valid_layouts = layouts[valid_mask]
    
    padded_layouts = F.pad(valid_layouts, (1, 1, 1, 1), mode='constant', value=0.0)
    diff_h = torch.abs(padded_layouts[:, 1:, :] - padded_layouts[:, :-1, :])
    diff_w = torch.abs(padded_layouts[:, :, 1:] - padded_layouts[:, :, :-1])
    
    tv_penalty = diff_h.sum(dim=(1, 2)).mean() + diff_w.sum(dim=(1, 2)).mean()
    
    # --- 4. Run Dynamic Slack Math (From experiment.py) ---
    macro_heights = layouts.max(dim=3)[0].sum(dim=2)
    macro_widths = layouts.max(dim=2)[0].sum(dim=2)
    
    valid_heights = macro_heights[valid_mask]
    valid_widths = macro_widths[valid_mask]
    
    expected_perimeters = (2.0 * valid_heights) + (2.0 * valid_widths)
    slack_value = expected_perimeters.mean()
    
    # --- 5. Print and Assert ---
    print(f"--- Cohesion Alignment Test ---")
    print(f"Macro 0 Expected Perimeter: {2*(10) + 2*(10)}")
    print(f"Macro 1 Expected Perimeter: {2*(5) + 2*(20)}")
    print(f"Batch Average Perimeter:    {(40 + 50) / 2}")
    print(f"-------------------------------")
    print(f"Calculated TV Penalty:      {tv_penalty.item()}")
    print(f"Calculated Slack Margin:    {slack_value.item()}")
    
    assert torch.isclose(tv_penalty, slack_value), "Mismatch detected!"
    print("\nSuccess! The Cohesion Penalty and Dynamic Slack are PERFECTLY aligned.")

if __name__ == "__main__":
    test_cohesion_alignment()