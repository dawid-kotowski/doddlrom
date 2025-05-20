# Master Project 1: Deep Learning-Based Reduced Order Modeling with PyMOR and PyTorch

## Overview

This repository contains a collection of deep learning-based reduced order models (ROMs) for parametric PDEs, implemented with PyTorch and integrated into the PyMOR ecosystem. The models are designed to approximate solution manifolds by learning low-dimensional representations and latent dynamics based on both geometric and physical parameters.

The framework supports training, evaluation, and benchmarking of various neural ROM architectures using precomputed data and leverages GPU acceleration for efficient training. It features modular implementations for selective model training and systematic storage of model checkpoints, training data, and benchmark results.

## Repository Structure

```
master_project_1/
├── examples/
│   ├── ex01/
│   │   ├── benchmarks/          # Performance metrics and results
│   │   ├── training_data/       # Preprocessed training datasets
│   │   ├── state_dicts/         # Saved model weights (state_dict)
│   │   └── module_training/     # Training scripts for different ROMs
│   └── ex02/
│       └── ...                  # Same structure as ex01
└── master_project_1/
    └── model_definitions.py     # Main PyTorch modules and training classes
```

## Mathematical Description

We aim to approximate a time-dependent solution \$u(t; \mu, \nu)\$ of a high-dimensional dynamical system in a reduced basis:

### Dynamic Orthogonal Decomposition (DOD\_DL)

Given

* \$\mu \in \Theta\$: geometric parameter
* \$\nu \in \Theta'\$: physical parameter
* \$t \in \[0,T]\$: time
* \$N\_A\$: ambient dimension
* \$N\$: reduced dimension

we construct $V: \Theta \times [0, T] \to \mathbb{R}^{N_A \times N}$ such that $u(t; \mu, \nu) \approx V(\mu, t) \cdot a(t; \mu, \nu),$ where \$V(\mu, t)\$ is learned by the **DOD\_DL** model.

### Coefficient Model (Coeff\_DOD\_DL)

We aim to learn a mapping $\Phi: \Theta \times \Theta' \times [0,T] \to \mathbb{R}^N, \quad (\mu, \nu, t) \mapsto a(t; \mu, \nu)$ via: $\Phi(\mu, \nu, t) := \sum_{i=1}^{m_0} \sum_{j=1}^{n_0} [\Phi_1(\mu, t)]_{ij} \cdot [\Phi_2(\nu, t)]_{ij},$ with \$\Phi\_1, \Phi\_2\$ as neural networks.

### Autoencoder-Based Model (AE\_DL)

The latent dynamics \$a(t; \mu, \nu)\$ are encoded by $\text{Encoder} : \Theta \times \Theta' \times [0,T] \to \mathbb{R}^N,$ with a nonlinear decoder reconstructing the high-dimensional solution. (Definition in subsequent modules)

## Available Models

* **DOD\_DL**: Learns time-dependent reduced basis \$V(\mu, t)\$.
* **Coeff\_DOD\_DL**: Learns latent coefficients \$a(t; \mu, \nu)\$ with bilinear interaction.
* **AE\_DL**: Autoencoder mapping for nonlinear projection (forthcoming).
* **Stationary Models**: Adaptations for time-independent problems (forthcoming).

## Training Workflow

Each `examples/exXX/module_training/` folder contains scripts to train and evaluate each ROM model:

1. Training data is stored in `examples/exXX/training_data/`
2. Training and validation split is controlled by `FetchReducedTrainAndValidSet`
3. Trained models are stored in `examples/exXX/state_dicts/`
4. Performance metrics are saved to `examples/exXX/benchmarks/`

## Dependencies

* Python >= 3.8
* PyTorch >= 2.0
* NumPy, tqdm
* [PyMOR](https://pymor.org/) >= 2023.1

## How to Run

Navigate into a specific example directory and run training, for example:

```bash
cd examples/ex01/module_training
python train_dod_dl.py
```

## Author & Acknowledgements

This repository is part of the author's Master's thesis under the supervision of Prof. Dr. Mario Ohlberger at the Institute for Applied Mathematics, University of Münster.

---

Please refer to each module's script for further implementation details and options.
