import os
import yaml
import torch
from tqdm import tqdm

# Import your models and evaluation functions
from models.cvae import ConditionalVAE 
from experiment import VAEXperiment
from dataset import VAEDataset
from core_engines.legalizer import Legalizer
from evaluation_metrics.overlap import calculate_exact_overlap
from evaluation_metrics.displacement import calculate_displacement
from evaluation_metrics.discretizer import extract_center_of_mass

def main():
    # 1. Setup Device and Config
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    with open("configs/cvae.yaml", 'r') as file:
        config = yaml.safe_load(file)
    with open("configs/data.yaml", "r") as f:
        data_config = yaml.safe_load(f)

    # Calculate dynamic channels (just like in run.py)
    max_macros = data_config['data_params']['num_macros'][1]
    config['model_params']['in_channels'] = max_macros + 1
    config['model_params']['out_channels'] = max_macros

    # 2. Load the Trained Model
    print("Loading model from checkpoint...")
    ckpt_path = config['eval_params']['ckpt_path']
    
    # Initialize the base architecture
    base_model = ConditionalVAE(**config['model_params'])
    
    # Inject the trained weights into the PyTorch Lightning wrapper
    experiment = VAEXperiment.load_from_checkpoint(
        ckpt_path, 
        vae_model=base_model, 
        params=config['exp_params']
    )
    
    # Lock the model into evaluation mode (disables dropout, freezes batchnorm)
    experiment.eval()
    experiment.to(device)

    # 3. Load the Test Data
    data = VAEDataset(**config["data_params"])
    data.setup()
    test_loader = data.test_dataloader()
    
    # Initialize your legalizer
    grid_h, grid_w = data_config['data_params']['grid_size']
    legalizer = Legalizer(grid_shape=(grid_h, grid_w))

    print("Beginning evaluation...")
    
    # 4. The Evaluation Loop
    total_overlap = 0.0
    total_displacement = 0.0
    successful_legalizations = 0
    failed_legalizations = 0
    
    with torch.no_grad():
        for batch_idx, (_, heat_maps, ground_truth_layouts) in enumerate(tqdm(test_loader)):
            heat_maps = heat_maps.to(device)
            ground_truth_layouts = ground_truth_layouts.to(device)
            batch_size = heat_maps.size(0)

            # A. Generate the continuous layouts
            continuous_layouts = experiment.model.sample(
                num_samples=batch_size, 
                current_device=device, 
                condition=heat_maps
            )

            # B. Inverse Rasterization (Continuous to Discrete)
            pre_legalized_boxes = extract_center_of_mass(continuous_layouts, ground_truth_layouts)

            # C. Metric 1: VAE Native Overlap
            batch_overlap = calculate_exact_overlap(pre_legalized_boxes)
            total_overlap += batch_overlap.sum().item()

            # D. Legalization
            # (Note: Because your legalizer currently takes a single chip at a time, 
            # you will need a quick list comprehension here to loop through the batch)
            post_legalized_boxes = []
            valid_pre_legalized_boxes = []
            for i in range(batch_size):
                try:
                    legal_chip = legalizer.make_legal(pre_legalized_boxes[i])
                    post_legalized_boxes.append(legal_chip)
                    valid_pre_legalized_boxes.append(pre_legalized_boxes[i])
                except:
                    failed_legalizations += 1

# E. Metric 2: Legalization Displacement (Only on successful chips)
            if len(post_legalized_boxes) > 0:
                post_legalized_boxes = torch.stack(post_legalized_boxes).to(device)
                valid_pre_legalized_boxes = torch.stack(valid_pre_legalized_boxes).to(device)

                batch_displacement = calculate_displacement(valid_pre_legalized_boxes, post_legalized_boxes)
                total_displacement += batch_displacement.sum().item()
                
                # Track how many chips we actually calculated displacement for
                successful_legalizations += len(post_legalized_boxes)

    # 5. Print Final Results
    num_test_samples = len(test_loader.dataset)
    failure_rate = (failed_legalizations / num_test_samples) * 100
    
    print("\n=== Final Evaluation Metrics ===")
    print(f"Average Pre-Legalization Overlap: {total_overlap / num_test_samples:.2f} sq units/chip")
    
    if successful_legalizations > 0:
        print(f"Average Legalizer Displacement:   {total_displacement / successful_legalizations:.2f} units/macro")
    else:
        print("Average Legalizer Displacement:   N/A (All chips failed legalization)")
        
    print(f"Legalization Failure Rate:        {failure_rate:.2f}% ({failed_legalizations} unsalvageable chips)")

if __name__ == "__main__":
    main()