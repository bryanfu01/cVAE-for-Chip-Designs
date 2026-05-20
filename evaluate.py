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
    experiment = VAEXperiment.load_from_checkpoint(ckpt_path, vae_model=base_model, params=config['exp_params'])
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

    print("Beginning evaluation...")
    
    total_overlap = 0.0
    total_displacement = 0.0
    total_thermal_mse = 0.0
    successful_legalizations = 0
    failed_legalizations = 0

    vis_original, vis_heatmap, vis_generated = None, None, None
    
    for batch_idx, (_, heat_maps, ground_truth_layouts, ground_truth_powers) in enumerate(tqdm(test_loader)):
        heat_maps = heat_maps.to(device)
        ground_truth_layouts = ground_truth_layouts.to(device)
        batch_size = heat_maps.size(0)

        # A. Sample initial latent vector
        z = torch.randn(batch_size, experiment.model.latent_dim, device=device)

        # B. Latent Space Optimization (Requires Gradients)
        if use_lso:
            z = optimize_latent_space(experiment.model, heat_maps, z, ground_truth_powers, lso_steps, lso_lr)

        # C. Discretization & Metrics (Freeze gradients to save RAM!)
        with torch.no_grad():
            flat_condition = heat_maps.view(batch_size, -1)
            continuous_layouts = experiment.model.decode(torch.cat([z, flat_condition], dim=1))
            
            pre_legalized_boxes = extract_center_of_mass(continuous_layouts, ground_truth_layouts)
            batch_overlap = calculate_exact_overlap(pre_legalized_boxes)
            total_overlap += batch_overlap.sum().item()

            post_legalized_boxes = []
            valid_pre_legalized_boxes = []
            
            for i in range(batch_size):
                try:
                    legal_chip = legalizer.make_legal(pre_legalized_boxes[i])
                    post_legalized_boxes.append(legal_chip)
                    valid_pre_legalized_boxes.append(pre_legalized_boxes[i])

                    # E. Thermal Performance Evaluation               
                    # Simulate the heat map of the legalized chip
                    simulated_heatmap = finite_solver.simulate(legal_chip.unsqueeze(0), ground_truth_powers)
                    target_heatmap = heat_maps[i].unsqueeze(0).unsqueeze(0)
                    
                    thermal_mse = torch.nn.functional.mse_loss(simulated_heatmap, target_heatmap)
                    total_thermal_mse += thermal_mse.item()

                    if vis_generated is None:
                        vis_generated = legal_chip
                        vis_original = ground_truth_layouts[i]
                        vis_heatmap = heat_maps[i]
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
    
    avg_overlap = total_overlap / num_test_samples
    avg_displacement = (total_displacement / successful_legalizations) if successful_legalizations > 0 else 0.0
    avg_thermal_mse = (total_thermal_mse / successful_legalizations) if successful_legalizations > 0 else 0.0

    # Format the results string
    results_str = f"""=== Final Evaluation Metrics (LSO: {use_lso}) ===
Average Pre-Legalization Overlap: {avg_overlap:.2f} sq units/chip
Average Legalizer Displacement:   {avg_displacement:.2f} units/macro
Average Thermal MSE:              {avg_thermal_mse:.6f}
Legalization Failure Rate:        {failure_rate:.2f}% ({failed_legalizations} unsalvageable chips)
"""

    # Print to console
    print("\n" + results_str)

    # Save to text file
    if txt_save_path:
        with open(txt_save_path, 'w') as f:
            f.write(results_str)
        print(f"Metrics saved to {txt_save_path}")

    if vis_generated is not None and img_save_path:
        plot_comparison(vis_original, vis_heatmap, vis_generated, grid_size=grid_w, save_path=img_save_path)

if __name__ == "__main__":
    main()