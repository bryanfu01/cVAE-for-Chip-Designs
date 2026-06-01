import torch
import torch.nn.functional as F

def test_area_mass_dilution():
    B, C, H, W = 2, 12, 64, 64
    
    # Simulate exactly 10 valid macros per batch, each perfectly 100 mass
    macro_powers = -1.0 * torch.ones(B, C)
    macro_powers[0, :10] = 0.5  # 10 valid in chip 1
    macro_powers[1, :10] = 0.5  # 10 valid in chip 2
    
    continuous_layouts = torch.zeros(B, C, H, W)
    for i in range(10):
        # Draw perfect 10x10 solid boxes (Mass = 100)
        continuous_layouts[0, i, 10:20, 10:20] = 1.0 
        continuous_layouts[1, i, 10:20, 10:20] = 1.0 
        
    print("--- Testing Area Penalty & Probe Calculations ---\n")
    
    # 1. The Flawed Probe Calculation (Currently in experiment.py)
    valid_mask_mult = (macro_powers != -1.0).view(B, C, 1, 1).float()
    valid_layouts_mult = continuous_layouts * valid_mask_mult
    diluted_mean = valid_layouts_mult.sum(dim=(2, 3)).mean().item()
    print(f"[CURRENT PROBE] Falsely reported mean mass: {diluted_mean:.2f}")
    print(f"                (Math: 10 valid / 12 total * 100 = 83.33)\n")
    
    # 2. The Upgraded Logic (Using your Boolean Indexing)
    valid_boolean_mask = (macro_powers != -1.0)
    true_valid_macros = continuous_layouts[valid_boolean_mask]
    true_mean = true_valid_macros.sum(dim=(1, 2)).mean().item()
    print(f"[TRUE METRIC]   Actual network mean mass: {true_mean:.2f}")
    print(f"                (The boolean mask drops the padding!)\n")
    
    # 3. Prove the Area Penalty Works
    target_areas = torch.full_like(true_valid_macros.sum(dim=(1, 2)), 100.0)
    area_loss = F.mse_loss(true_valid_macros.sum(dim=(1, 2)), target_areas)
    print(f"[AREA ENGINE]   MSE Loss for perfectly drawn boxes: {area_loss.item():.4f}")

if __name__ == '__main__':
    test_area_mass_dilution()