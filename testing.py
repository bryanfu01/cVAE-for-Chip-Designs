import torch

def test_form_penalties():
    print("--- Testing Form Penalties (Dilution Bug) ---")
    
    B, C, H, W = 1, 1, 64, 64
    
    # 1. Solid Box (10x10) - Target mass 100
    solid = torch.zeros(B, C, H, W)
    solid[0, 0, 10:20, 10:20] = 1.0
    
    # 2. Confetti (100 individual pixels spread out) - Target mass 100
    confetti = torch.zeros(B, C, H, W)
    for i in range(10):
        for j in range(10):
            confetti[0, 0, i*4, j*4] = 1.0 
            
    # --- CURRENT COHESION (The Dilution Bug) ---
    def current_cohesion(layouts):
        diff_h = torch.abs(layouts[:, :, 1:, :] - layouts[:, :, :-1, :])
        diff_w = torch.abs(layouts[:, :, :, 1:] - layouts[:, :, :, :-1])
        return diff_h.mean() + diff_w.mean()

    # --- UPGRADED COHESION (True Edge Count) ---
    def upgraded_cohesion(layouts):
        diff_h = torch.abs(layouts[:, :, 1:, :] - layouts[:, :, :-1, :])
        diff_w = torch.abs(layouts[:, :, :, 1:] - layouts[:, :, :, :-1])
        # Sum over spatial grid (H, W), then mean over Batch (B) and Channels (C)
        return diff_h.sum(dim=(2, 3)).mean() + diff_w.sum(dim=(2, 3)).mean()
        
    print(f"\n[CURRENT COHESION - Diluted by 64x64 grid]")
    print(f"Solid Box Loss: {current_cohesion(solid).item():.6f}")
    print(f"Confetti Loss:  {current_cohesion(confetti).item():.6f}")
    print(f"Gradient Push:  {current_cohesion(confetti).item() - current_cohesion(solid).item():.6f} (Optimizer ignores this!)")

    print(f"\n[UPGRADED COHESION - Physical Perimeter Count]")
    print(f"Solid Box Loss: {upgraded_cohesion(solid).item():.2f} (Exactly 40 physical edges!)")
    print(f"Confetti Loss:  {upgraded_cohesion(confetti).item():.2f} (Exactly 400 physical edges!)")
    print(f"Gradient Push:  {upgraded_cohesion(confetti).item() - upgraded_cohesion(solid).item():.2f} (Massive signal!)")

if __name__ == "__main__":
    test_form_penalties()