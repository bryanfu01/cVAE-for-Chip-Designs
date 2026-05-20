import torch
from core_engines.soft_drcs import SoftDRC

def optimize_latent_space(model, condition, z, ground_truth_powers=None, lso_steps=50, lr=0.05, drc_evaluator=None):
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
        drc_metrics = drc_evaluator(continuous_layouts, condition)
        total_penalty = drc_metrics['total_drc_loss']
        
        # Backpropagate to z
        total_penalty.backward()
        optimizer.step()
        
    return z.detach()