# Generative Chip Layout Design in Differentiable EDA

**Physics-Informed Conditional VAEs for Thermally-Aware Macro Placement**

This repository contains the official PyTorch implementation of the framework proposed in the paper **"Generative Chip Layout Design in Differentiable EDA."** This project tackles the computational bottlenecks of modern Electronic Design Automation (EDA) by introducing a Deep Generative Model (DGM) capable of zero-shot chip macro placement. By restricting the environment to a 2D grid and leveraging a differentiable finite difference solver, this framework trains a Conditional Variational Autoencoder (CVAE) to instantly generate continuous, thermally-aware chip layouts that adhere to strict Design Rule Checks (DRCs) via proxy physics penalties.

## Key Features

* **Amortized Layout Generation:** Replaces iterative analytical solvers with a single neural network forward pass, drastically reducing inference time.
* **Soft DRC Physics Engine:** Translates discrete design rules (Overlap, Area, Thermal compliance) and structural rules (Sharpness, Cohesion) into continuous, differentiable loss penalties.
* **Bounded Dual Ascent (BDA):** Dynamically balances opposing physical forces (e.g., Thermal vs. Overlap) during training by updating Lagrangian dual variables, capped to prevent gradient explosion.
* **Latent Space Optimization (LSO):** An inference-time refinement step that performs gradient descent directly on the latent vector $z$ to sculpt the probability mass into legally compliant structures before discretization.
* **Custom 2D Finite Difference Solver:** A highly parallelized, PyTorch-native solver used for dataset generation and final thermal evaluation.

---

## Repository Structure

├── configs/                  # YAML configuration files
│   ├── cvae.yaml             # Model architecture and Soft DRC hyperparameters
│   ├── data.yaml             # Synthetic dataset generation parameters
│   └── eval.yaml             # Latent Space Optimization (LSO) parameters
├── core_engines/             # Physics and rule-checking engines
│   ├── finite_solver.py      # 2D Thermal Heat Equation solver
│   ├── legalizer.py          # Greedy discrete legalization algorithm
│   └── soft_drcs.py          # Differentiable physics penalties
├── data_generation/          # Dataset creation scripts
│   └── data_generator.py     # Multiprocessing pipeline for synthetic layouts
├── evaluation_metrics/       # Metric calculation and visualization
│   ├── discretizer.py        # Center-of-mass continuous-to-discrete extraction
│   ├── displacement.py       # Calculates Euclidean displacement post-legalization
│   ├── overlap.py            # Exact macro overlap calculation
│   └── visualize.py          # Generates Target vs. Simulated visual comparisons
├── models/                   # Neural network architectures
│   ├── base.py               # Abstract VAE class
│   └── cvae.py               # Conditional VAE with deterministic mean decoding
├── run.py                    # PyTorch Lightning training orchestrator
├── run_data.py               # Dataset generation execution script
├── evaluate.py               # Evaluation and LSO execution script
├── experiment.py             # LightningModule containing BDA and training loops
├── latent_optimization.py    # Inference-time optimization over latent vector z
└── requirements.txt          # Python package dependencies