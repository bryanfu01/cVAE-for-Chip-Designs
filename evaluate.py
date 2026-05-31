import os
import yaml
import torch
from tqdm import tqdm

from models.cvae import ConditionalVAE 
from experiment import VAEXperiment
from dataset import VAEDataset
from core_engines.legalizer import Legalizer
from core_engines.finite_solver import FiniteDifferenceSolver
from evaluation_metrics.overlap import calculate_exact_overlap
from evaluation_metrics.displacement import calculate_displacement
from evaluation_metrics.discretizer import extract_center_of_mass
from evaluation_metrics.visualize import plot_comparison
from latent_optimization import optimize_latent_space
from core_engines.soft_drcs import SoftDRC

def main():
    # 1. Setup Device and Configs
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    with open("configs/cvae.yaml", 'r') as file_1: config = yaml.safe_load(file_1)
    with open("configs/data.yaml", "r") as file_2: data_config = yaml.safe_load(file_2)
    with open("configs/eval.yaml", "r") as file_3: eval_config = yaml.safe_load(file_3)

    max_macros = data_config['data_params']['num_macros'][1]
    config['model_params']['in_channels'] = max_macros + 1
    config['model_params']['out_channels'] = max_macros

    # Extract Eval Parameters
    ckpt_path = config["trainer_params"]["resume_ckpt_path"]
    img_save_path = eval_config['saving_params']['img_path']
    txt_save_path = eval_config['saving_params']['result_path']
    use_lso = eval_config['latent_optim']['use_lso']
    lso_steps = eval_config['latent_optim']['steps']
    lso_lr = eval_config['latent_optim']['lr']

    # 2. Load Model & Data
    base_model = ConditionalVAE(**config['model_params'])
    experiment = VAEXperiment.load_from_checkpoint(ckpt_path, 
                                                   vae_model=base_model, 
                                                   params=config['exp_params'], 
                                                   soft_drc_params=config['soft_drc_params'])
    experiment.eval()
    experiment.to(device)

    data = VAEDataset(**config["data_params"])
    data.setup()
    test_loader = data.test_dataloader()
    
    grid_h, grid_w = data_config['data_params']['grid_size']
    legalizer = Legalizer(grid_shape=(grid_h, grid_w))
    
    # Initialize the finite solver for the Thermal Evaluation
    finite_solver = FiniteDifferenceSolver(grid_shape=(grid_h, grid_w), 
                                           k_conductivity=data_config['finite_solver_params']['conductivity'],
                                           dx=eval(data_config['finite_solver_params']['resolution']),
                                           tolerance=data_config['finite_solver_params']['tolerance'],
                                           iterations=data_config['finite_solver_params']['iterations'])
    
    drc_evaluator = SoftDRC(
        overlap_weight=config['soft_drc_params']['overlap_weight'], 
        area_weight=config['soft_drc_params']['area_weight'], 
        thermal_weight=config['soft_drc_params']['thermal_weight'],
        sharpness_weight=config['soft_drc_params']['sharpness_weight'],
        target_area=config['soft_drc_params']['target_area'],
        cohesion_weight=config['soft_drc_params']['cohesion_weight']
    )
    
    print("Beginning evaluation...")
    
    total_overlap = 0.0
    total_displacement = 0.0
    total_thermal_mse = 0.0
    successful_legalizations = 0
    failed_legalizations = 0
    latent_variance = 0
    total_mass = 0

    # Output configurable number of results (mfu)
    num_comparisons = eval_config['saving_params'].get('num_comparisons', 5)
    comparison_samples = []
    #vis_original, vis_heatmap, vis_generated = None, None, None
    # End (mfu)
    
    for batch_idx, (rasterized_layouts, heat_maps, ground_truth_layouts, ground_truth_powers) in enumerate(tqdm(test_loader)):
        heat_maps = heat_maps.to(device)
        ground_truth_layouts = ground_truth_layouts.to(device)
        batch_size = heat_maps.size(0)

        # A. Sample initial latent vector
        z = torch.randn(batch_size, experiment.model.latent_dim, device=device)

        # B. Latent Space Optimization (Requires Gradients)
        if use_lso:
            z = optimize_latent_space(
                model=experiment.model, 
                condition=heat_maps, 
                z=z, 
                ground_truth_powers=ground_truth_powers, 
                lso_steps=lso_steps, 
                lr=lso_lr,
                drc_evaluator=drc_evaluator,
                regulariztion=config["exp_params"]["kld_weight"]
            )
        # C. Discretization & Metrics (Freeze gradients to save RAM!)
        with torch.no_grad():
            flat_condition = heat_maps.view(batch_size, -1)
            continuous_layouts = experiment.model.decode(torch.cat([z, flat_condition], dim=1))
            
            # --- HEATMAP SENSITIVITY DIAGNOSTIC (runs once) --- (mfu)
            if batch_idx == 0:
                z_fixed = z[0:1]
                heatmap_1 = heat_maps[0:1]
                flat_heatmap_1 = heatmap_1.view(1, -1)
                heatmap_2 = heat_maps[1:2]
                flat_heatmap_2 = heatmap_2.view(1, -1)
                heatmap_zeros = torch.zeros_like(heatmap_1)
                flat_heatmap_zeros = heatmap_zeros.view(1, -1)
                heatmap_ones = torch.ones_like(heatmap_1)
                flat_heatmap_ones = heatmap_ones.view(1, -1)

                diff_real = (experiment.model.decode(torch.cat([z_fixed, flat_heatmap_1], dim=1)) -
                             experiment.model.decode(torch.cat([z_fixed, flat_heatmap_2], dim=1))).abs().mean().item()
                diff_zeros = (experiment.model.decode(torch.cat([z_fixed, flat_heatmap_1], dim=1)) -
                              experiment.model.decode(torch.cat([z_fixed, flat_heatmap_zeros], dim=1))).abs().mean().item()
                diff_ones = (experiment.model.decode(torch.cat([z_fixed, flat_heatmap_1], dim=1)) -
                             experiment.model.decode(torch.cat([z_fixed, flat_heatmap_ones], dim=1))).abs().mean().item()

                print(f"\n=== HEATMAP SENSITIVITY DIAGNOSTIC ===")
                print(f"Same z, different real heatmaps:  {diff_real:.6f}")
                print(f"Same z, real vs zero heatmap:     {diff_zeros:.6f}")
                print(f"Same z, real vs all-ones heatmap: {diff_ones:.6f}")
                print(f"(Values near 0 = decoder ignoring heatmap)")
                print(f"(Values > 0.01 = decoder responding to heatmap)")
                print("=======================================\n")

                # --- DIAGNOSTIC 1: z=zeros vs z=random ---
                z_zeros = torch.zeros(1, experiment.model.latent_dim, device=device)
                layout_z_zeros = experiment.model.decode(torch.cat([z_zeros, flat_heatmap_1], dim=1))
                layout_z_random = experiment.model.decode(torch.cat([z_fixed, flat_heatmap_1], dim=1))
                diff_z = (layout_z_zeros - layout_z_random).abs().mean().item()
                print(f"=== Z=ZEROS DIAGNOSTIC ===")
                print(f"Same heatmap, z=zeros vs z=random: {diff_z:.6f}")
                print(f"(Near 0 = decoder ignoring z)")
                print("===========================\n")

                # --- DIAGNOSTIC 2: Encoder mu values ---
                encoder_input = torch.cat([rasterized_layouts[0:1].to(device), heat_maps[0:1]], dim=1)
                mu, log_var = experiment.model.encode(encoder_input)
                print(f"=== ENCODER OUTPUT DIAGNOSTIC ===")
                print(f"mu min/max:      {mu.min().item():.4f} / {mu.max().item():.4f}")
                print(f"mu mean abs:     {mu.abs().mean().item():.4f}")
                print(f"log_var min/max: {log_var.min().item():.4f} / {log_var.max().item():.4f}")
                print(f"std min/max:     {torch.exp(0.5*log_var).min().item():.4f} / {torch.exp(0.5*log_var).max().item():.4f}")
                print("=================================\n")

                # --- DIAGNOSTIC 3: z sensitivity on center of mass ---
                print(f"=== Z SENSITIVITY DIAGNOSTIC ===")
                for trial in range(5):
                    z_trial = torch.randn(1, experiment.model.latent_dim, device=device)
                    layout_trial = experiment.model.decode(torch.cat([z_trial, flat_heatmap_1], dim=1))
                    combined = layout_trial.sum(dim=1)  # [1, H, W]
                    h_grid = torch.arange(combined.shape[1], device=device).float()
                    w_grid = torch.arange(combined.shape[2], device=device).float()
                    total_mass = combined.sum() + 1e-8
                    cy = (combined * h_grid.view(1,-1,1)).sum() / total_mass
                    cx = (combined * w_grid.view(1,1,-1)).sum() / total_mass
                    print(f"Trial {trial}: center of mass = ({cx.item():.1f}, {cy.item():.1f})")
                print("================================\n")
                # --- END DIAGNOSTIC --- (mfu)
                num_layouts = 100
                layout_1 = ground_truth_layouts[0:1]
                variance_layouts = base_model.sample(num_samples=num_layouts, condition=heatmap_1)
                variance_com = extract_center_of_mass(continuous_layout=variance_layouts, ground_truth_macros=layout_1)
                latent_variance = torch.var(variance_com, dim=0)

            total_mass = continuous_layouts.sum(dim=(2, 3))

            pre_legalized_boxes = extract_center_of_mass(continuous_layouts, ground_truth_layouts)
            batch_overlap = calculate_exact_overlap(pre_legalized_boxes)
            total_overlap += batch_overlap.sum().item()

            post_legalized_boxes = []
            valid_pre_legalized_boxes = []
            
            for i in range(batch_size):
                try:
                    legal_chip = legalizer.make_legal(pre_legalized_boxes[i].squeeze())
                    post_legalized_boxes.append(legal_chip)
                    valid_pre_legalized_boxes.append(pre_legalized_boxes[i])

                    # E. Thermal Performance Evaluation               
                    # Simulate the heat map of the legalized chip
                    chip_powers = ground_truth_powers[i].tolist()
                    simulated_heatmap = finite_solver.simulate(legal_chip, chip_powers)
                    s_max = simulated_heatmap.max()
                    s_min = simulated_heatmap.min()
                    if s_max > s_min:
                        simulated_heatmap = (simulated_heatmap - s_min) / (s_max - s_min)

                    target_heatmap = heat_maps[i].unsqueeze(0)
                    
                    thermal_mse = torch.nn.functional.mse_loss(simulated_heatmap, target_heatmap)
                    """
                    print(f"\n--- Batch {batch_idx} Debug ---")
                    print(f"Target Sum: {target_heatmap.sum().item():.4f} | Sim Sum: {simulated_heatmap.sum().item():.4f}")
                    print(f"Target Max: {target_heatmap.max().item():.4f} | Sim Max: {simulated_heatmap.max().item():.4f}")
                    print(f"MSE Value:  {thermal_mse.item()}")
                    """

                    total_thermal_mse += thermal_mse.item()

                    # output configurable number of comparisons (mfu)
                    if len(comparison_samples) < num_comparisons:
                        comparison_samples.append({
                        'generated_chip_layout': legal_chip,
                        'generated_heat_map': simulated_heatmap,
                        'original_chip_layout': ground_truth_layouts[i],
                        'original_heatmap': heat_maps[i]
                    })
                    #if vis_generated is None:
                    #    vis_generated = legal_chip
                    #    vis_original = ground_truth_layouts[i]
                    #    vis_heatmap = heat_maps[i]
                    # End (mfu)
                except ValueError:
                    failed_legalizations += 1

            if len(post_legalized_boxes) > 0:
                post_legalized_boxes = torch.stack(post_legalized_boxes).to(device)
                valid_pre_legalized_boxes = torch.stack(valid_pre_legalized_boxes).to(device)
                batch_displacement = calculate_displacement(valid_pre_legalized_boxes, post_legalized_boxes)
                total_displacement += batch_displacement.sum().item()
                successful_legalizations += len(post_legalized_boxes)

    # 5. Compile and Save Final Results
    num_test_samples = len(test_loader.dataset)
    failure_rate = (failed_legalizations / num_test_samples) * 100
    
    mass_fidelity = total_mass/num_test_samples
    avg_overlap = total_overlap / num_test_samples
    avg_displacement = (total_displacement / successful_legalizations) if successful_legalizations > 0 else 0.0
    avg_thermal_mse = (total_thermal_mse / successful_legalizations) if successful_legalizations > 0 else 0.0

    # Format the results string
    results_str = f"""=== Final Evaluation Metrics (LSO: {use_lso}) ===
Average Pre-Legalization Overlap: {avg_overlap:.2f} sq units/chip
Average Legalizer Displacement:   {avg_displacement:.2f} units/macro
Average Thermal MSE:              {avg_thermal_mse:.6f}
Legalization Failure Rate:        {failure_rate:.2f}% ({failed_legalizations} unsalvageable chips)
Latent Space Variance Test:    {latent_variance:.2f}
Mass Fidelity Test:               {mass_fidelity:.2f}
"""

    # Print to console
    print("\n" + results_str)

    # Save to text file
    if txt_save_path:
        with open(txt_save_path, 'w') as f:
            f.write(results_str)
        print(f"Metrics saved to {txt_save_path}")

    # output configurable number of comparisons (mfu)
    for idx, sample in enumerate(comparison_samples): 
        save_path = img_save_path.replace('.png', f'_{idx}.png')
        plot_comparison(
            sample['original_heat_map'],
            sample['generated_heat_map'],
            sample['original_chip_layout'],
            sample['generated_chip_layout'],
            grid_size=grid_w,
            save_path=save_path
        )
    #if vis_generated is not None and img_save_path:
    #    plot_comparison(vis_original, vis_heatmap, vis_generated, grid_size=grid_w, save_path=img_save_path)
    # End (mfu)

if __name__ == "__main__":
    main()