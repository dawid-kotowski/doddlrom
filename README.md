# Master Project 1: Deep Learning-Based Reduced Order Modeling with PyMOR and PyTorch

## Overview

This repository offers a suite of deep learning-based reduced order models (ROMs) for parametric partial differential equations (PDEs). These models are developed in PyTorch and integrated with [PyMOR](https://pymor.org/), enabling efficient, data-driven model reduction that accounts for both geometric and physical parameter dependencies.

The framework is built for modularity and GPU-accelerated execution. It supports selective training of components, consistent handling of datasets and checkpoints, and comprehensive evaluation through benchmark logging.

## Repository Structure

```
master_project_1/
├── examples/
│   ├── ex01/
│   │   ├── benchmarks/          # Quantitative evaluations and plots
│   │   ├── training_data/       # Parameter snapshots and latent states
│   │   ├── state_dicts/         # Trained model weights
│   │   └── module_training/     # Training scripts and validation routines
│   └── ex02/
│       └── ...                  # Identical structure for a second benchmark
└── master_project_1/
    └── dod_dl_rom.py     # Core PyTorch modules and architecture definitions
```

## Mathematical Framework

The objective is to approximate the high-fidelity, time-dependent solution $u_h(t; \mu, \nu) \in \mathbb{R}^{N_h}$ of a parametric dynamical system using a learned surrogate:

* $\mu \in \Theta$: geometric parameter
* $\nu \in \Theta'$: physical parameter
* $t \in [0, T]$: time
* $\mathbb{A} \in \mathbb{R}^{N_h \times N_A}$: POD map
* $N_A$: quasi full-order model dimension (prior reduction by POD)
* $N \ll N_A$: reduced dimension
* $n \ll N_A$: true latent dimension

### Deep Orthogonal Decomposition (DOD-DL)

The DOD-DL model seeks a time- and geometry-dependent orthonormal basis $V(\mu, t) \in \mathbb{R}^{N_A \times N}$ such that

$$
    u(t; \mu, \nu) \approx V(\mu, t) \cdot \Phi(t; \mu, \nu),
$$

where the low-dimensional representation $\Phi(t; \mu, \nu) \in \mathbb{R}^N$ encodes the parametric time evolution on the latent space.

### Linear Coefficient Model for the DOD-DL

This model directly approximates the latent space dynamic:

$$
    \Phi: \Theta \times \Theta' \times [0, T] \to \mathbb{R}^N, \quad (\mu, \nu, t) \mapsto a(t; \mu, \nu).
$$

It is structured via a bilinear neural interaction:

$$
    \Phi(\mu, \nu, t) = \sum_{i=1}^{m_0} \sum_{j=1}^{n_0} [\Phi_1(\mu, t)]_{ij} \cdot [\Phi_2(\nu, t)]_{ij},
$$

where $\Phi_1$ and $\Phi_2$ represent learned embeddings of geometric and physical parameters.

### Autoencoder-Based Coefficient Model for the DOD-DL

In AE-DL, the full state is reconstructed from a latent code via:

* **Encoder**: maps $u_{N}(\mu, \nu, t)$ to latent representation $u_{n}(t; \mu, \nu)$
* **Latent Coefficient Model**: maps $(\mu, \nu, t)$ to the correct latent representation $u_n(t; \mu, \nu)$
* **Decoder**: reconstructs the high-dimensional state from latent representation

This approach allows for general nonlinear compression and decoding of PDE solution trajectories.

### Proper Orthogonal Decomposition (POD-DL)

This model is directly to be found in the Paper [POD-DL-ROM: enhancing deep learning-based reduced order models for nonlinear parametrized PDEs by proper orthogonal decomposition](https://arxiv.org/abs/2101.11845) by S.Fresca and A.Manzoni. It is essentially the same workflow as the the Model above, but omiting the DOD projection onto the dimension $N$, i.e.

* **Encoder**: maps $u_{N_A}(\mu, \nu, t)$ to latent representation $u_{n}(t; \mu, \nu)$
* **Latent Coefficient Model**: maps $(\mu, \nu, t)$ to the correct latent representation $u_n(t; \mu, \nu)$
* **Decoder**: reconstructs the high-dimensional state from latent representation

### Stationary solution enhanced CoLoRA (CoLoRA-DL)

To discussed...

## Training Pipeline

Each `examples/exXX/module_training/` directory includes standalone scripts with a uniform pipeline:

1. Load preprocessed data from `training_data/`
2. Use `FetchReducedTrainAndValidSet` for data partitioning
3. Train model and store weights in `state_dicts/`
4. Save quantitative results to `benchmarks/`

## Dependencies

* Python ≥ 3.8
* PyTorch ≥ 2.0
* NumPy, tqdm
* PyMOR ≥ 2023.1

Install all dependencies via:

```bash
pip install -r requirements.txt
```

## Getting Started

To train a ROM model, navigate to an example directory and execute a training script:

```bash
cd examples/ex01/module_training
python train_dod_dl.py
```

## Author & Acknowledgements

This repository supports the Master's thesis of the author, conducted under the supervision of Prof. Dr. Mario Ohlberger at the Institute for Applied Mathematics, University of Münster.

---

Further architectural and numerical implementation details can be found in the docstrings and module comments accompanying each training and model definition script. 
