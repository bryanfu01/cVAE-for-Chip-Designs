import torch

def optimize_latent_space(model, condition, z, ground_truth_powers, lso_steps=50, lr=0.05):
    """
    Refines the latent vector z using gradients from differentiable physics penalties.
    """
    # 1. Detach z and explicitly tell PyTorch we want to train it
    z = z.clone().detach().requires_grad_(True)
    
    # 2. Setup an optimizer specifically for this single vector
    optimizer = torch.optim.Adam([z], lr=lr)
    
    # Flatten condition once
    B = condition.size(0)
    flat_condition = condition.view(B, -1)

    for step in range(lso_steps):
        optimizer.zero_grad()
        
        # Decode the current z
        decoder_input = torch.cat([z, flat_condition], dim=1)
        continuous_layout = model.decode(decoder_input)
        
        # --- CALCULATE DIFFERENTIABLE LOSSES HERE ---
        # Example: Soft Overlap Penalty (You will insert your Soft DRC math here)
        # We want to push z in a direction that minimizes overlapping probability clouds
        density_sum = continuous_layout.sum(dim=1) # Sum all macros
        overlap_penalty = torch.nn.functional.relu(density_sum - 1.0).sum()
        
        # Backpropagate to z
        overlap_penalty.backward()
        optimizer.step()
        
    return z.detach()