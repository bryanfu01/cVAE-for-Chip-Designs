import torch
from core_engines.soft_drcs import SoftDRC

def optimize_latent_space(model, condition, z, ground_truth_powers=None, lso_steps=50, lr=0.05, drc_evaluator=None, regulariztion=0.00025, batch_idx=None):
    """
    Refines the latent vector z using gradients from the unified SoftDRC physics penalties.
    """
    # 1. Initialize the shared physics evaluator if one isn't provided
    if drc_evaluator is None:
        drc_evaluator = SoftDRC()

    # 2. Detach z and explicitly tell PyTorch we want to train it
    z = z.clone().detach().requires_grad_(True)
    
    # 3. Setup an optimizer specifically for this single vector
    optimizer = torch.optim.Adam([z], lr=lr)
    
    # Flatten condition once
    B = condition.size(0)
    flat_condition = condition.view(B, -1)

    for step in range(lso_steps):
        optimizer.zero_grad()
        
        # Decode the current z
        decoder_input = torch.cat([z, flat_condition], dim=1)
        continuous_layouts = model.decode(decoder_input)
        
        # --- UNIFIED DIFFERENTIABLE PHYSICS ---
        # Call the exact same SoftDRC module used in experiment.py!
        drc_metrics = drc_evaluator(continuous_layouts, condition, ground_truth_powers)
        drc_penalty = drc_metrics['total_drc_loss']
        
        
        # NEW: Calculate Latent Regularization (Anchor z to standard normal)
        # We multiply by your exact kld_weight from the yaml!
        z_penalty =  regulariztion * 0.5 * torch.sum(z ** 2)
        total_penalty = drc_penalty + z_penalty
        # Backpropagate to z
        total_penalty.backward()

        if batch_idx == 0 and step % 10 == 0:
            print(f"LSO Step {step:02d} | Physics Penalty: {drc_penalty.item():.4f} | "
                  f"Z-Reg Penalty: {z_penalty.item():.4f} | Z-StdDev: {z.std().item():.4f}")

        # PROBE 4: Gradient Flow Check (Only on the first step of the first batch)
        if step == 0:
            grad_magnitude = z.grad.abs().mean().item()
            print(f"LSO Step 0 - Latent Gradient Magnitude: {grad_magnitude:.6f}")
            if grad_magnitude == 0.0:
                print("WARNING: DEAD GRADIENT. The decoder is completely disconnected from z!")
        
        optimizer.step()
        
    return z.detach()