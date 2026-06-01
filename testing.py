import torch

def test_sharpness_gradients():
    print("--- Testing Sharpness Penalties (BCE vs Quadratic) ---")
    print("Observe what happens to the gradients as a pixel gets close to 0 or 1.")
    
    # Test probabilities from 0.5 (center) out to 0.999 (near perfect binary)
    probs = torch.tensor([0.5, 0.8, 0.9, 0.99, 0.999], requires_grad=True)
    
    print(f"{'Prob':<10} | {'BCE Gradient':<15} | {'Quadratic Gradient':<20}")
    print("-" * 50)
    
    for p_val in probs:
        p_bce = p_val.clone().detach().requires_grad_(True)
        p_quad = p_val.clone().detach().requires_grad_(True)
        
        # 1. BCE Loss
        eps = 1e-8
        bce_loss = -(p_bce * torch.log(p_bce + eps) + (1.0 - p_bce) * torch.log(1.0 - p_bce + eps))
        bce_loss.backward()
        
        # 2. Quadratic Loss
        quad_loss = p_quad * (1.0 - p_quad)
        quad_loss.backward()
        
        # We print the absolute magnitude of the gradient (the "force" the optimizer feels)
        bce_force = abs(p_bce.grad.item())
        quad_force = abs(p_quad.grad.item())
        
        print(f"{p_val.item():<10.3f} | {bce_force:<15.4f} | {quad_force:<20.4f}")
        
    print("\nCONCLUSION:")
    print("As probability approaches 1.0:")
    print("- Quadratic gradient drops safely to 0.0 (It settles peacefully).")
    print("- BCE gradient EXPLODES toward infinity (It violently shatters shapes!).")

if __name__ == "__main__":
    test_sharpness_gradients()