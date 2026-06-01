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
            drc_metrics = self.soft_drc_evaluator(recons, heat_maps, powers)

            warmup_epochs = self.soft_drc_params.get('warmup_epochs', 30)

            if warmup_epochs != 0:
                warmup_factor = min(1.0, self.current_epoch / warmup_epochs)
            else:
                warmup_factor = 1
            scaled_drc_loss = drc_metrics['total_drc_loss'] * warmup_factor + (1 - warmup_factor) * drc_metrics['Soft_Sharpness_Loss']

            # PROBE 3: Gradient Balance (Prints once per epoch)
            if batch_idx == 0:
                base_loss = self.soft_drc_params.get('vanilla_weight') * train_loss['loss'].item()
                raw_drc = drc_metrics['total_drc_loss'].item()
                raw_sharpness = drc_metrics['Soft_Sharpness_Loss'].item()
                print(f"=== PROBE 3: EPOCH {self.current_epoch} LOSS BALANCE ===")
                print(f"Base VAE Loss:    {base_loss:.4f}")
                print(f"Raw Soft DRC:     {raw_drc:.4f}")
                print(f"Warmup Multiplier: {warmup_factor:.4f}")
                print(f"Effective DRC:    {(raw_drc * warmup_factor + (1 - warmup_factor) * raw_sharpness):.4f}\n")

                # Mask layout and multiply by heat
                actual_thermal_exposure = (valid_layouts * heat_maps).sum().item()
                print(f"Total Heat Exposure: {actual_thermal_exposure:.2f} (Should decay over epochs)")

            train_loss['loss'] = self.soft_drc_params.get('vanilla_weight') * train_loss['loss'] + scaled_drc_loss
            
            # Merge the isolated metrics for TensorBoard tracking
            train_loss.update({k: v for k, v in drc_metrics.items() if k != 'total_drc_loss'})

            train_loss['DRC_Warmup_Factor'] = torch.tensor(warmup_factor)
            train_loss.update({k: (v * warmup_factor) for k, v in drc_metrics.items() if k != 'total_drc_loss'})

        self.log_dict({key: val.item() for key, val in train_loss.items()}, sync_dist=True)

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
            drc_metrics = self.soft_drc_evaluator(recons, heat_maps, powers)
            
            # Add the raw physics penalty directly to the total val_loss
            val_loss['loss'] = self.soft_drc_params.get('vanilla_weight', 100.0) * val_loss['loss'] + drc_metrics['total_drc_loss']
            
            # Merge the individual tracking metrics (Overlap, Area, Thermal)
            val_loss.update({k: v for k, v in drc_metrics.items() if k != 'total_drc_loss'})

        self.log_dict({f"val_{key}": val.item() for key, val in val_loss.items()}, sync_dist=True)
        
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

