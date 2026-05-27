import torch
import torch.nn as nn
import torch.nn.functional as F

class SoftDRC(nn.Module):
    def __init__(self, 
                 overlap_weight: float = 50.0, 
                 area_weight: float = 20.0, 
                 thermal_weight: float = 10.0,
                 sharpness_weight: float = 20.0,
                 target_area: float = 100.0,
                 cohesion_weight: float = 20.0): 
        """
        Differentiable Physics Penalties for Continuous Chip Layouts.
        """
        super(SoftDRC, self).__init__()
        self.overlap_weight = overlap_weight
        self.area_weight = area_weight
        self.thermal_weight = thermal_weight
        self.sharpness_weight = sharpness_weight
        self.cohesion_weight = cohesion_weight

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
        
        # The expected probability mass sum for a single macro channel
        self.target_area = target_area

    def _calculate_overlap_penalty(self, continuous_layouts: torch.Tensor, macro_powers: torch.Tensor) -> torch.Tensor:
        B, C, H, W = continuous_layouts.shape
        
        # 1. Create a mask to ignore padded channels
        valid_mask = (macro_powers != -1.0).view(B, C, 1, 1).float().to(self.device)
        
        valid_layouts = continuous_layouts * valid_mask
        
        density_sum = valid_layouts.sum(dim=1) 
        overlap_error = F.relu(density_sum - 1.0)
        
        return (overlap_error ** 2).mean()

    def _calculate_area_penalty(self, continuous_layouts: torch.Tensor, macro_powers: torch.Tensor) -> torch.Tensor:
        B, C, H, W = continuous_layouts.shape
        macro_masses = continuous_layouts.sum(dim=(2, 3)) 
        
        # 1. Target area should be 100.0 for real macros, but 0.0 for padded macros!
        target_areas = torch.where(macro_powers != -1.0, 
                                   torch.full_like(macro_masses, self.target_area), 
                                   torch.zeros_like(macro_masses))
        
        return F.mse_loss(macro_masses, target_areas)
    
    def _calculate_thermal_penalty(self, continuous_layouts: torch.Tensor, target_heatmaps: torch.Tensor, macro_powers: torch.Tensor) -> torch.Tensor:
        """
        Calculates the Physics Force (Thermal Dispersion).
        Penalizes layouts that place continuous macro density in cold regions,
        weighted heavily by the actual power output of the specific macro.
        """
        B, C, H, W = continuous_layouts.shape
        # Normalize the heatmap to [0, 1] for stable gradient scaling
        heatmap_max = target_heatmaps.view(B, -1).max(dim=1)[0].view(B, 1, 1, 1)
        heatmap_min = target_heatmaps.view(B, -1).min(dim=1)[0].view(B, 1, 1, 1)
        normalized_heatmaps = (target_heatmaps - heatmap_min) / (heatmap_max - heatmap_min + 1e-8)
        
        # Calculate the inverse heatmap (1.0 = cold, 0.0 = hot)
        inverse_heatmaps = 1.0 - normalized_heatmaps
        
        # NEW: Reshape the power tensor for spatial broadcasting (B, C) -> (B, C, 1, 1)
        # We replace the -1 padding with 0 so padded macros don't contribute to the penalty
        safe_powers = torch.where(macro_powers == -1.0, torch.zeros_like(macro_powers), macro_powers)
        power_weights = safe_powers.view(B, C, 1, 1)

        # Multiply layout density by its specific power, then by the inverse heatmap
        # High-power macros in cold spots will generate massive gradient penalties!
        thermal_penalty = (continuous_layouts.to(self.device) * power_weights.to(self.device)) * inverse_heatmaps.to(self.device)
        
        return thermal_penalty.mean()
    
    def _calculate_sharpness_penalty(self, continuous_layouts: torch.Tensor, macro_powers: torch.Tensor) -> torch.Tensor:
        B, C, H, W = continuous_layouts.shape
        valid_mask = (macro_powers != -1.0).view(B, C, 1, 1).float().to(self.device)
        valid_layouts = continuous_layouts * valid_mask
        
        # Penalizes values near 0.5. The penalty drops to 0 at exactly 0.0 or 1.0.
        eps = 1e-8
        bce = -(valid_layouts * torch.log(valid_layouts + eps) + 
                (1.0 - valid_layouts) * torch.log(1.0 - valid_layouts + eps))
        return bce.mean()
    
    def _calculate_cohesion_penalty(self, continuous_layouts: torch.Tensor, macro_powers: torch.Tensor) -> torch.Tensor:
        """
        Total Variation Loss. Penalizes scattered pixels by summing the physical 
        edges of the probability distributions. Forces macros to clump into solid shapes.
        """
        B, C, H, W = continuous_layouts.shape
        valid_mask = (macro_powers != -1.0).view(B, C, 1, 1).float().to(self.device)
        valid_layouts = continuous_layouts * valid_mask

        # Calculate differences between adjacent pixels (finding the edges)
        diff_h = torch.abs(valid_layouts[:, :, 1:, :] - valid_layouts[:, :, :-1, :])
        diff_w = torch.abs(valid_layouts[:, :, :, 1:] - valid_layouts[:, :, :, :-1])

        return diff_h.mean() + diff_w.mean()

    def forward(self, continuous_layouts: torch.Tensor, target_heatmaps: torch.Tensor, macro_powers: torch.Tensor) -> dict:
        
        # Make sure to pass macro_powers to ALL THREE functions now!
        overlap_loss = self._calculate_overlap_penalty(continuous_layouts, macro_powers.to(self.device))
        area_loss = self._calculate_area_penalty(continuous_layouts, macro_powers.to(self.device))
        thermal_loss = self._calculate_thermal_penalty(continuous_layouts, target_heatmaps, macro_powers.to(self.device))
        sharpness_loss = self._calculate_sharpness_penalty(continuous_layouts, macro_powers.to(self.device))
        cohesion_loss = self._calculate_cohesion_penalty(continuous_layouts, macro_powers.to(self.device))

        # 2. Apply hyperparameter weights
        weighted_overlap = self.overlap_weight * overlap_loss
        weighted_area = self.area_weight * area_loss
        weighted_thermal = self.thermal_weight * thermal_loss
        weighted_sharpness = self.sharpness_weight * sharpness_loss
        weighted_cohesion = self.cohesion_weight * cohesion_loss

        # 3. Sum into the total Soft DRC loss
        total_drc_loss = weighted_overlap + weighted_area + weighted_thermal + weighted_sharpness + weighted_cohesion

        # 4. Return dictionary formatted for TensorBoard logging
        return {
            'total_drc_loss': total_drc_loss,
            'Soft_Overlap_Loss': weighted_overlap.detach(),
            'Soft_Area_Loss': weighted_area.detach(),
            'Soft_Thermal_Loss': weighted_thermal.detach(),
            'Soft_Sharpness_Loss': weighted_sharpness.detach(),
            'Soft_Cohesion_Loss': weighted_cohesion.detach()
        }