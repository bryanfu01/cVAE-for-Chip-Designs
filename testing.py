import torch
import torch.nn.functional as F

def test_physics_magnitudes():
    # 1. Setup Mock Grid [Batch, Channels, Height, Width]
    B, C, H, W = 1, 2, 64, 64
    continuous_layouts = torch.zeros((B, C, H, W))
    
    # 2. Create Two 10x10 Macros (Area = 100 each)
    # We will stack them PERFECTLY on top of each other
    continuous_layouts[0, 0, 10:20, 10:20] = 1.0
    continuous_layouts[0, 1, 10:20, 10:20] = 1.0
    
    macro_powers = torch.tensor([[1.0, 1.0]])

    # --- 3. Test OVERLAP Penalty (Using spatial .sum) ---
    valid_mask = (macro_powers != -1.0).view(B, C, 1, 1).float()
    valid_layouts = continuous_layouts * valid_mask

    density_sum = valid_layouts.sum(dim=1)  # Where they overlap, density is 2.0
    overlap_error = F.relu(density_sum - 1.0) # ReLU(2.0 - 1.0) = 1.0 error per pixel
    
    # Sum over the 100 overlapping pixels
    overlap_penalty = (overlap_error ** 2).sum(dim=(1, 2)).mean()

    # --- 4. Test THERMAL Penalty (Using spatial .sum) ---
    # We simulate a "worst-case" scenario where the 10x10 region has maximum thermal penalty (1.0)
    inverse_heatmaps = torch.zeros((B, 1, H, W))
    inverse_heatmaps[0, 0, 10:20, 10:20] = 1.0 

    power_weights = macro_powers.view(B, C, 1, 1)
    
    # Penalty per pixel = 1.0 (Layout) * 1.0 (Power) * 1.0 (Thermal) = 1.0
    thermal_calc = (continuous_layouts * power_weights) * inverse_heatmaps
    
    # Sum over the 100 pixels AND both channels
    thermal_penalty = thermal_calc.sum(dim=(1, 2, 3)).mean()

    # --- 5. Print Results ---
    print(f"--- Magnitude Alignment Test ---")
    print(f"Overlap Penalty (2 stacked 10x10 macros):     {overlap_penalty.item():.2f}")
    print(f"Thermal Penalty (2 macros in worst-case zone): {thermal_penalty.item():.2f}")
    print(f"--------------------------------")
    print(f"Magnitude Ratio (Thermal / Overlap):           {thermal_penalty.item() / overlap_penalty.item():.2f}x")
    print(f"(Perfectly aligned within O(100) magnitude!)")

if __name__ == "__main__":
    test_physics_magnitudes()