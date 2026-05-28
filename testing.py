import torch
from core_engines.soft_drcs import SoftDRC

def test_area_penalty():
    print("--- Testing Area Penalty ---")
    # Initialize DRC engine
    drc = SoftDRC(area_weight=1.0, target_area=100.0)
    
    # Create a batch of 2 chips: 
    # Chip 0: Macro mass is 50 (Penalty expected: (50-100)^2 = 2500)
    # Chip 1: Macro mass is 100 (Penalty expected: 0)
    B, C, H, W = 2, 1, 64, 64
    continuous_layouts = torch.zeros(B, C, H, W)
    
    # Fill Chip 0 with 50 pixels of mass (value 1.0)
    continuous_layouts[0, 0, :50, 0] = 1.0 
    
    # Fill Chip 1 with 100 pixels of mass (value 1.0)
    continuous_layouts[1, 0, :10, :10] = 1.0
    
    # Define macro powers: -1.0 means padding, 1.0 means active
    macro_powers = torch.tensor([[-1.0], [-1.0]])
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    
    # Calculate penalty
    penalty = drc._calculate_area_penalty(continuous_layouts, macro_powers.to(device))
    
    print(f"Computed MSE Penalty: {penalty.item():.4f}")
    
    # Verify expectations
    expected_penalty = ( (50 - 100)**2 + (100 - 100)**2 ) / 2
    if torch.isclose(penalty, torch.tensor(float(expected_penalty))):
        print("SUCCESS: Area penalty math is correct.")
    else:
        print(f"FAILURE: Expected {expected_penalty}, got {penalty.item()}")

# Run the test
if __name__ == "__main__":
    test_area_penalty()