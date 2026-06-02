import os
import math
import torch
from torch import optim
from models import BaseVAE
from models.types_ import *
import pytorch_lightning as pl
from torchvision import transforms
import torchvision.utils as vutils
from torch.utils.data import DataLoader
from core_engines.soft_drcs import SoftDRC


class VAEXperiment(pl.LightningModule):

    def __init__(self,
                 vae_model: BaseVAE,
                 params: dict,
                 soft_drc_params: dict) -> None:
        super(VAEXperiment, self).__init__()

        self.model = vae_model
        self.params = params
        self.soft_drc_params = soft_drc_params
        self.curr_device = None
        self.hold_graph = False
        try:
            self.hold_graph = self.params['retain_first_backpass']
        except:
            pass
        self.soft_drc_evaluator = SoftDRC(
            overlap_weight=self.soft_drc_params.get('overlap_weight', 50.0),
            area_weight=self.soft_drc_params.get('area_weight', 20.0),
            thermal_weight=self.soft_drc_params.get('thermal_weight', 10.0),
            sharpness_weight=self.soft_drc_params.get('sharpness_weight', 100.0),
            target_area=self.soft_drc_params.get('target_area', 20.0), # E.g., for a 10x10 footprint
            cohesion_weight=self.soft_drc_params.get('cohesion_weight', 20.0)
            )

        self.alm_lr = 0.01  # Alpha: How fast the weights are allowed to grow
        self.ema_decay = 0.99 # Smoothing factor for the moving average

        # Initialize the Trainable Dual Variables (Lambdas)
        # requires_grad=False because we update them manually, not via Adam
        self.lambda_overlap = torch.nn.Parameter(torch.tensor(1.0), requires_grad=False)
        self.lambda_area = torch.nn.Parameter(torch.tensor(1.0), requires_grad=False)
        self.lambda_sharpness = torch.nn.Parameter(torch.tensor(1.0), requires_grad=False)
        self.lambda_cohesion = torch.nn.Parameter(torch.tensor(1.0), requires_grad=False)
        self.lambda_thermal = torch.nn.Parameter(torch.tensor(1.0), requires_grad=False)

        # Register Buffers for the EMAs (so PyTorch handles device placement automatically)
        self.register_buffer('ema_overlap', torch.tensor(1.0))
        self.register_buffer('ema_area', torch.tensor(1.0))
        self.register_buffer('ema_sharpness', torch.tensor(1.0))
        self.register_buffer('ema_cohesion', torch.tensor(1.0))
        self.register_buffer('ema_thermal', torch.tensor(1.0))

    def forward(self, input: Tensor, condition: Tensor) -> Tensor:
        # Equivalent to self.model.forward(input), but pytorch works better this way for backprop
        return self.model(input, condition)

    def training_step(self, batch, batch_idx):
        layouts, heat_maps, _, powers = batch
        self.curr_device = layouts.device

        if batch_idx == 0 and self.current_epoch == 0:
            print("\n=== PROBE 1: DATA PIPELINE ===")
            print(f"Heatmap Min/Max: {heat_maps.min().item():.4f} / {heat_maps.max().item():.4f}")
            print(f"Layouts Shape:   {layouts.shape}")
            print(f"Sample Powers:   {powers[0].tolist()}")
            print("==============================\n")

        results = self.forward(input=layouts, condition=heat_maps)

        if batch_idx == 0:
            recons_probe = results[0]
            print(f"\n=== PROBE 2: EPOCH {self.current_epoch} RAW OUTPUT ===")
            print(f"Raw Prob Min/Max: {recons_probe.min().item():.4f} / {recons_probe.max().item():.4f}")
            
            # Check the average "mass" of the continuous macros. 
            # If target_area is 100, this should ideally climb toward 100 over time.
            B, C, H, W = recons_probe.shape
            valid_boolean_mask = (powers != -1.0)
            valid_layouts = recons_probe[valid_boolean_mask]
            mean_mass = valid_layouts.sum(dim=(1, 2)).mean().item()
            print(f"Mean Macro Mass:  {mean_mass:.2f} (True Mass)")
            
            # Debugging: output distribution and sparsity (mfu)
            counts = torch.histc(recons_probe, bins=10, min=0.0, max=1.0)
            print(f"Output distribution: {[f'{c.item():.0f}' for c in counts]}")
            input_nonzero = (results[1] > 0.5).float().mean().item()
            recons_nonzero = (recons_probe > 0.5).float().mean().item()
            print(f"Input nonzero:   {input_nonzero:.6f}")
            print(f"Recons nonzero:  {recons_nonzero:.6f}")  

            gray_pixels = ((valid_layouts > 0.1) & (valid_layouts < 0.9)).sum().item()
            total_valid_pixels = valid_layouts.numel() 
            
            gray_ratio = (gray_pixels / total_valid_pixels) * 100 if total_valid_pixels > 0 else 0.0
            print(f"Gray Zone Pixels: {gray_ratio:.2f}% (Should drop to 0%)")

            # End (mfu)

        train_loss = self.model.loss_function(*results,
                                              M_N=self.params['kld_weight'], 
                                              batch_idx=batch_idx)
                                              
        # Debugging: loss decomposition (mfu)
        if batch_idx == 0:
            recons_probe = results[0]
            input_probe = results[1]
            B, C, H, W = recons_probe.shape

            # --- Macro Count Stats (fixed, printed once) ---
            if self.current_epoch == 0:
                real_macro_counts = (powers != -1.0).sum(dim=1).float()
                print(f"--- Macro Count Stats (epoch 0 only) ---")
                print(f"Min real macros:  {real_macro_counts.min().item():.0f}")
                print(f"Max real macros:  {real_macro_counts.max().item():.0f}")
                print(f"Mean real macros: {real_macro_counts.mean().item():.2f}")
                print(f"Total channels:   {C}")
                print(f"Avg padded channels: {(C - real_macro_counts.mean().item()):.2f}")
                for count in range(C + 1):
                    n = (real_macro_counts == count).sum().item()
                    if n > 0:
                        print(f"  {count} macros: {n} samples")

            # --- MSE Breakdown ---
            valid_mask = (powers != -1.0).view(B, C, 1, 1).float().to(self.device)
            padding_mask = (powers == -1.0).view(B, C, 1, 1).float().to(self.device)

            real_recons = recons_probe * valid_mask
            real_input = input_probe * valid_mask
            real_pixel_count = valid_mask.sum()
            mse_real_channels = ((real_recons - real_input) ** 2).sum() / real_pixel_count

            pad_recons = recons_probe * padding_mask
            pad_input = input_probe * padding_mask
            pad_pixel_count = padding_mask.sum()
            mse_pad_channels = ((pad_recons - pad_input) ** 2).sum() / (pad_pixel_count + 1e-8)

            foreground_mask = (input_probe > 0.5).float()
            background_mask = ((input_probe <= 0.5).float()) * valid_mask

            fg_count = foreground_mask.sum()
            bg_count = background_mask.sum()

            mse_foreground = ((recons_probe - input_probe) ** 2 * foreground_mask).sum() / (fg_count + 1e-8)
            mse_background = ((recons_probe - input_probe) ** 2 * background_mask).sum() / (bg_count + 1e-8)

            print(f"--- MSE Breakdown (Epoch {self.current_epoch}) ---")
            print(f"Overall MSE:          {train_loss['Reconstruction_Loss'].item():.6f}")
            print(f"MSE real channels:    {mse_real_channels.item():.6f}  ({real_pixel_count.item():.0f} pixels)")
            print(f"MSE padded channels:  {mse_pad_channels.item():.6f}  ({pad_pixel_count.item():.0f} pixels)")
            print(f"MSE foreground only:  {mse_foreground.item():.6f}  ({fg_count.item():.0f} macro pixels)")
            print(f"MSE background only:  {mse_background.item():.6f}  ({bg_count.item():.0f} background pixels)")
            print(f"Foreground/Background pixel ratio: {(fg_count / (bg_count + 1e-8)).item():.4f}")

            # Check for fake macro generation — compare active output channels vs real macro count
            real_macro_counts = (powers != -1.0).sum(dim=1).float()  # [B] ground truth count
    
            # A channel is "active" if the model output has meaningful mass
            channel_masses = recons_probe.sum(dim=(2, 3))  # [B, C]
            active_threshold = 10.0  # channels with mass > 10 are considered active
            active_counts = (channel_masses > active_threshold).float().sum(dim=1)  # [B]
    
            print(f"--- Fake Macro Check ---")
            print(f"Avg real macros:    {real_macro_counts.mean().item():.2f}")
            print(f"Avg active channels:{active_counts.mean().item():.2f}")
            fake_macros = (active_counts - real_macro_counts).clamp(min=0)
            print(f"Avg fake macros:    {fake_macros.mean().item():.2f}")
            print(f"Max fake macros:    {fake_macros.max().item():.0f}")
            print(f"Samples with fakes: {(fake_macros > 0).sum().item()} / {B}")
            
            recon = train_loss['Reconstruction_Loss'].item()
            kld = train_loss['KLD'].item()
            gamma = train_loss['Gamma'].item()
            print(f"Recon Loss:       {recon:.6f}")
            print(f"KLD (free bits):  {kld:.4f}")
            print(f"Gamma:            {gamma:.6f}")
            print(f"KLD contribution: {gamma * abs(kld):.6f}")
            print(f"Recon/KLD ratio:  {recon / (gamma * abs(kld) + 1e-8):.2f}")
            mu = results[2]
            log_var = results[3]
            kld_per_dim = -0.5 * (1 + log_var - mu**2 - log_var.exp())  # [B, latent_dim]
            dims_above_lambda = (kld_per_dim > 0.5).float().mean().item()
            kld_per_dim_mean = kld_per_dim.mean().item()
            kld_per_dim_max = kld_per_dim.max().item()
            print(f"--- Free Bits Diagnostic ---")
            print(f"KLD per dim (mean): {kld_per_dim_mean:.4f}")
            print(f"KLD per dim (max):  {kld_per_dim_max:.4f}")
            print(f"Dims above lambda (0.5): {dims_above_lambda:.4f}  ({dims_above_lambda*128:.1f} / 128 dims)")
        # End (mfu)
        
        if self.soft_drc_params.get('use_soft_drc', False):
            recons = results[0]
            drc_metrics = self.soft_drc_evaluator(recons, heat_maps, powers, true_layouts=layouts)

            # 1. Extract raw constraint violations (Detach them so ALM updates don't flow backward into the VAE)
            raw_overlap = drc_metrics['Soft_Overlap_Loss'].detach()
            raw_area = drc_metrics['Soft_Area_Loss'].detach()
            raw_thermal = drc_metrics['Soft_Thermal_Loss'].detach()
            raw_sharpness_ = drc_metrics['Soft_Sharpness_Loss'].detach()
            raw_cohesion = drc_metrics['Soft_Cohesion_Loss'].detach()

            # 2. Update the Exponential Moving Averages (EMA)
            self.ema_overlap = (self.ema_decay * self.ema_overlap) + ((1 - self.ema_decay) * raw_overlap)
            self.ema_area = (self.ema_decay * self.ema_area) + ((1 - self.ema_decay) * raw_area)
            self.ema_thermal = (self.ema_decay * self.ema_thermal) + ((1 - self.ema_decay) * raw_thermal)
            self.ema_sharpness = (self.ema_decay * self.ema_sharpness) + ((1 - self.ema_decay) * raw_sharpness_)
            self.ema_cohesion = (self.ema_decay * self.ema_cohesion) + ((1 - self.ema_decay) * raw_cohesion)

            # --- THE FULLY DYNAMIC COHESION MARGIN ---
            # 'layouts' shape: (Batch, 4, Max_Macros). Index 2 is Height, Index 3 is Width.
            # 'powers' shape: (Batch, Max_Macros). -1.0 means padded.
            
            valid_boolean_mask = (powers != -1.0)
            
           # Extract the actual Heights and Widths of only the valid macros
            macro_heights = layouts.max(dim=3)[0].sum(dim=2)  # Shape: [B, C]
            macro_widths = layouts.max(dim=2)[0].sum(dim=2)   # Shape: [B, C]

            # 2. Filter out the padded channels using the mask
            valid_heights = macro_heights[valid_boolean_mask]
            valid_widths = macro_widths[valid_boolean_mask]
            
            # Theoretical Total Variation (Perimeter) for a solid macro is 2H + 2W
            expected_perimeters = (2.0 * valid_heights) + (2.0 * valid_widths)
            
            # Because soft_drcs.py averages across all valid macros, our margin is the mean!
            # We add a tiny 0.5 buffer to account for continuous probability blurring at the edges.
            dynamic_cohesion_margin = expected_perimeters.mean().item() + 0.5
            
            # Calculate the true violation
            cohesion_violation = raw_cohesion - dynamic_cohesion_margin
            # -----------------------------------------

            # 3. Normalized Dual Ascent Step (Update the Lambdas)
            # Only increase lambda if there is actually a violation!
            if raw_overlap > 0.1:
                self.lambda_overlap.data += self.alm_lr * (raw_overlap / (self.ema_overlap + 1e-5))
            if raw_area > 0.1:
                self.lambda_area.data += self.alm_lr * (raw_area / (self.ema_area + 1e-5))
            if raw_thermal > 0.1:
                self.lambda_thermal.data += self.alm_lr * (raw_thermal / (self.ema_thermal + 1e-5))
            if raw_sharpness_ > 0.1:
                self.lambda_sharpness.data += self.alm_lr * (raw_sharpness_ / (self.ema_sharpness + 1e-5))
            if cohesion_violation > 0.1:
                self.lambda_cohesion.data += self.alm_lr * (cohesion_violation / (self.ema_cohesion + 1e-5))


            self.log('Lambda_Overlap', self.lambda_overlap)
            self.log('Lambda_Area', self.lambda_area)
            self.log('Lambda_Thermal', self.lambda_thermal)
            self.log('Lambda_Sharpness', self.lambda_sharpness)
            self.log('Lambda_Cohesion', self.lambda_cohesion)

            # 1. Create the differentiable Cohesion Violation tensor
            cohesion_graph_violation = torch.relu(drc_metrics['Soft_Cohesion_Loss'] - dynamic_cohesion_margin)

            # 2. EMA LOSS NORMALIZATION
            # Divide by the EMA to force every constraint's base magnitude to exactly ~1.0
            norm_overlap = drc_metrics['Soft_Overlap_Loss'] / (self.ema_overlap.detach() + 1e-5)
            norm_area = drc_metrics['Soft_Area_Loss'] / (self.ema_area.detach() + 1e-5)
            norm_thermal = drc_metrics['Soft_Thermal_Loss'] / (self.ema_thermal.detach() + 1e-5)
            norm_sharpness = drc_metrics['Soft_Sharpness_Loss'] / (self.ema_sharpness.detach() + 1e-5)
            norm_cohesion = cohesion_graph_violation / (self.ema_cohesion.detach() + 1e-5)

            # 3. Apply the ALM Lambda Weights to the Normalized Losses
            total_physics_loss = (self.lambda_overlap * norm_overlap) + \
                                 (self.lambda_area * norm_area) + \
                                 (self.lambda_thermal * norm_thermal) + \
                                 (self.lambda_sharpness * norm_sharpness) + \
                                 (self.lambda_cohesion * norm_cohesion)
        
            # PROBE 3: Gradient Balance (Prints once per epoch)
            if batch_idx == 0:
                base_loss = self.soft_drc_params.get('vanilla_weight', 1) * train_loss['loss'].item()
                raw_drc = drc_metrics['total_drc_loss'].item()
                raw_sharpness = drc_metrics['Soft_Sharpness_Loss'].item()
                print(f"=== PROBE 3: EPOCH {self.current_epoch} LOSS BALANCE ===")
                print(f"Base VAE Loss:    {base_loss:.4f}")
                print(f"Raw Soft DRC:     {raw_drc:.4f}")
               # ADD .item() TO ALL OF THESE:
                print(f"Overlap Weight:   {self.lambda_overlap.item():.4f} (Normalized: {norm_overlap.item():.4f})")
                print(f"Area Weight:      {self.lambda_area.item():.4f} (Normalized: {norm_area.item():.4f})")
                print(f"Thermal Weight:   {self.lambda_thermal.item():.4f} (Normalized: {norm_thermal.item():.4f})")
                print(f"Sharpness Weight: {self.lambda_sharpness.item():.4f} (Normalized: {norm_sharpness.item():.4f})")
                print(f"Cohesion Weight:  {self.lambda_cohesion.item():.4f} (Normalized: {norm_cohesion.item():.4f})")
                print(f"Effective DRC:    {total_physics_loss.item():.4f}\n")

                # Mask layout and multiply by heat
                actual_thermal_exposure = (valid_layouts * heat_maps).sum().item()
                print(f"Total Heat Exposure: {actual_thermal_exposure:.2f} (Should decay over epochs)")

            train_loss['loss'] = self.soft_drc_params.get('vanilla_weight', 1) * train_loss['loss'] + total_physics_loss
            
           # 7. Merge the RAW metrics for TensorBoard tracking
            train_loss.update({k: v.detach() for k, v in drc_metrics.items() if k != 'total_drc_loss'})

            # 8. Track the EFFECTIVE (Lambda-Scaled) losses
            train_loss['ALM_Effective_Overlap'] = (self.lambda_overlap * drc_metrics['Soft_Overlap_Loss']).detach()
            train_loss['ALM_Effective_Area'] = (self.lambda_area * drc_metrics['Soft_Area_Loss']).detach()
            # ... (add others if you wish to track them)

        self.log_dict({key: val.item() if isinstance(val, torch.Tensor) else val for key, val in train_loss.items()}, sync_dist=True)

        return train_loss['loss']

    def validation_step(self, batch, batch_idx):
        layouts, heat_maps, _, powers = batch
        self.curr_device = layouts.device

        results = self.forward(layouts, condition=heat_maps)
        val_loss = self.model.loss_function(*results,
                                            M_N=1.0, 
                                            batch_idx=batch_idx)
        
        if self.soft_drc_params.get('use_soft_drc', False):
            recons = results[0]
            drc_metrics = self.soft_drc_evaluator(recons, heat_maps, powers, true_layouts=layouts)

            # --- THE FULLY DYNAMIC COHESION MARGIN ---
            # 'layouts' shape: (Batch, 4, Max_Macros). Index 2 is Height, Index 3 is Width.
            # 'powers' shape: (Batch, Max_Macros). -1.0 means padded.
            
            valid_boolean_mask = (powers != -1.0)
            
            # Extract the actual Heights and Widths of only the valid macros
            macro_heights = layouts.max(dim=3)[0].sum(dim=2)  # Shape: [B, C]
            macro_widths = layouts.max(dim=2)[0].sum(dim=2)   # Shape: [B, C]

            # 2. Filter out the padded channels using the mask
            valid_heights = macro_heights[valid_boolean_mask]
            valid_widths = macro_widths[valid_boolean_mask]
            
            # Theoretical Total Variation (Perimeter) for a solid macro is 2H + 2W
            expected_perimeters = (2.0 * valid_heights) + (2.0 * valid_widths)
            
            # Because soft_drcs.py averages across all valid macros, our margin is the mean!
            # We add a tiny 0.5 buffer to account for continuous probability blurring at the edges.
            dynamic_cohesion_margin = expected_perimeters.mean().item() + 0.5

            cohesion_graph_violation = torch.relu(drc_metrics['Soft_Cohesion_Loss'] - dynamic_cohesion_margin)

            # 2. EMA LOSS NORMALIZATION
            # Divide by the EMA to force every constraint's base magnitude to exactly ~1.0
            norm_overlap = drc_metrics['Soft_Overlap_Loss'] / (self.ema_overlap.detach() + 1e-5)
            norm_area = drc_metrics['Soft_Area_Loss'] / (self.ema_area.detach() + 1e-5)
            norm_thermal = drc_metrics['Soft_Thermal_Loss'] / (self.ema_thermal.detach() + 1e-5)
            norm_sharpness = drc_metrics['Soft_Sharpness_Loss'] / (self.ema_sharpness.detach() + 1e-5)
            norm_cohesion = cohesion_graph_violation / (self.ema_cohesion.detach() + 1e-5)

            val_physics_loss = (self.lambda_overlap * norm_overlap) + \
                                 (self.lambda_area * norm_area) + \
                                 (self.lambda_thermal * norm_thermal) + \
                                 (self.lambda_sharpness * norm_sharpness) + \
                                 (self.lambda_cohesion * norm_cohesion)

            # Add the ALM penalty directly to the total val_loss
            val_loss['loss'] = self.soft_drc_params.get('vanilla_weight', 1) * val_loss['loss'] + val_physics_loss
            
            # Merge the raw tracking metrics
            val_loss.update({k: v.detach() for k, v in drc_metrics.items() if k != 'total_drc_loss'})

        self.log_dict({f"val_{key}": val.item() if isinstance(val, torch.Tensor) else val for key, val in val_loss.items()}, sync_dist=True)
        
    def on_validation_end(self) -> None:
        self.sample_images()
        
    def sample_images(self):
        # Get sample reconstruction image            
        test_input, test_label, _, _ = next(iter(self.trainer.datamodule.test_dataloader()))
        test_input = test_input.to(self.curr_device)
        test_label = test_label.to(self.curr_device)

#         test_input, test_label = batch
        recons = self.model.generate(test_input, condition = test_label)
        recons_vis = recons.data.sum(dim=1, keepdim=True)
        vutils.save_image(recons_vis,
                          os.path.join(self.logger.log_dir , 
                                       "Reconstructions", 
                                       f"recons_{self.logger.name}_Epoch_{self.current_epoch}.png"),
                          normalize=True,
                          nrow=12)

        try:
            batch_size = test_input.size(0)
            samples = self.model.sample(num_samples=batch_size,
                                        current_device=self.curr_device,
                                        condition = test_label)
            samples_vis = samples.cpu().data.sum(dim=1, keepdim=True)
            vutils.save_image(samples_vis,
                              os.path.join(self.logger.log_dir , 
                                           "Samples",      
                                           f"{self.logger.name}_Epoch_{self.current_epoch}.png"),
                              normalize=True,
                              nrow=12)
        except Warning:
            pass

    def configure_optimizers(self):

        optimizer = optim.Adam(self.model.parameters(),
                               lr=self.params['LR'],
                               weight_decay=self.params['weight_decay'])
        
        return optimizer

