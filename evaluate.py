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

    print("\n--- Extracted BDA Weights from Checkpoint ---")
    lam_overlap = experiment.lambda_overlap.item()
    lam_area = experiment.lambda_area.item()
    lam_thermal = experiment.lambda_thermal.item()
    lam_sharpness = experiment.lambda_sharpness.item()
    lam_cohesion = experiment.lambda_cohesion.item()
    
    print(f"Lambda Overlap:   {lam_overlap:.4f}")
    print(f"Lambda Area:      {lam_area:.4f}")
    print(f"Lambda Thermal:   {lam_thermal:.4f}")
    print(f"Lambda Sharpness: {lam_sharpness:.4f}")
    print(f"Lambda Cohesion:  {lam_cohesion:.4f}")
    print("---------------------------------------------\n")

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
    
    # --- NEW: Initialize SoftDRC with EFFECTIVE BDA Weights ---
    # Effective Weight = (Base YAML Weight) * (Learned BDA Lambda)
    drc_evaluator = SoftDRC(
        overlap_weight=config['soft_drc_params']['overlap_weight'] * lam_overlap, 
        area_weight=config['soft_drc_params']['area_weight'] * lam_area, 
        thermal_weight=config['soft_drc_params']['thermal_weight'] * lam_thermal,
        sharpness_weight=config['soft_drc_params']['sharpness_weight'] * lam_sharpness,
        target_area=config['soft_drc_params']['target_area'],
        cohesion_weight=config['soft_drc_params']['cohesion_weight'] * lam_cohesion
    )
    
    print("Beginning evaluation...")
    
    total_overlap = 0.0
    total_displacement = 0.0
    total_thermal_mse = 0.0
    total_mass = 0.0
    total_valid_macros = 0
    successful_legalizations = 0
    failed_legalizations = 0
    latent_variance = 0.0

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
                regulariztion=config["exp_params"]["kld_weight"],
                batch_idx=batch_idx
            )
        # C. Discretization & Metrics (Freeze gradients to save RAM!)
        with torch.no_grad():
            flat_condition = heat_maps.view(batch_size, -1)
            continuous_layouts = experiment.model.decode(torch.cat([z, flat_condition], dim=1))
            
            # --- 1. LATENT SPATIAL VARIANCE TEST (Runs once on the first batch) ---
            if batch_idx == 0:
                num_layouts = 100
                # Expand dimensions to prevent Concatenation and Discretization crashes
                repeated_heatmap = heat_maps[0:1].repeat(num_layouts, 1, 1, 1)
                repeated_layout = ground_truth_layouts[0:1].repeat(num_layouts, 1, 1)
                
                variance_layouts = experiment.model.sample(num_samples=num_layouts, 
                                                           current_device=device, 
                                                           condition=repeated_heatmap)
                variance_com = extract_center_of_mass(variance_layouts, repeated_layout)
                
                # Calculate average spatial variance strictly on valid (unpadded) macros
                valid_mask = repeated_layout[0, 0, :] != -1
                x_var = torch.var(variance_com[:, 0, valid_mask], dim=0).mean().item()
                y_var = torch.var(variance_com[:, 1, valid_mask], dim=0).mean().item()
                latent_variance = (x_var + y_var) / 2.0
            
            # --- 2. MASS FIDELITY TEST (Accumulates across all batches) ---
            valid_power_mask = (ground_truth_powers != -1.0).view(batch_size, -1, 1, 1).to(device)
            batch_valid_layouts = continuous_layouts * valid_power_mask
            total_mass += batch_valid_layouts.sum().item()
            total_valid_macros += (ground_truth_powers != -1.0).sum().item()

            # --- 3. PRE-LEGALIZATION OVERLAP ---
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
                        'generated_heatmap': simulated_heatmap,
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
    # 5. Compile and Save Final Results
    num_test_samples = len(test_loader.dataset)
    failure_rate = (failed_legalizations / num_test_samples) * 100
    
    avg_overlap = total_overlap / num_test_samples
    avg_displacement = (total_displacement / successful_legalizations) if successful_legalizations > 0 else 0.0
    avg_thermal_mse = (total_thermal_mse / successful_legalizations) if successful_legalizations > 0 else 0.0
    
    # Calculate mass strictly per valid macro (Target = 100.0)
    mass_fidelity = (total_mass / total_valid_macros) if total_valid_macros > 0 else 0.0

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
            sample['original_heatmap'],
            sample['generated_heatmap'],
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