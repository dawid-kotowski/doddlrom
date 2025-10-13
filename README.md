# DODDL-ROM: Deep Orthogonal Decomposition Deep Learning ROMs

**Deep Learning-Based Reduced Order Modeling Framework for Parametric PDEs**

---

## Repository Structure

```
doddlrom/
├── examples/                # Example problems and their data/training outputs
│   ├── ex01/ ...            # Example 1 (e.g. 2D inviscid Burgers equation)
│   ├── ex02/ ...            # Example 2 (analogous structure to ex01)
│   └── ex03/ ...            # Example 3 (analogous structure to ex01)
├── core/                    # Core library code for ROM models
│   ├── reduced_order_models.py    # PyTorch architectures and trainer classes 
│   └── configs/parameters.py     # All hyperparameters for each example
├── utils/                   # Utility and convenience scripts
│   ├── send_code.sh         # Syncs the code to a remote server (HPC) via SSH
│   └── fetch_work.sh        # Fetches result files from remote $WORK directory
├── environment.yml          # Conda environment specification (dependencies)
└── setup.py                 # Installation script (pip install configuration)
```

---

## Installation Guide

To set up the Python environment, use the provided Conda environment file:

```
conda env create -f environment.yml   # creates env (with PyTorch, PyMOR, etc.)
conda activate doddlrom              # activate the new environment
```

This will install all required packages (Python ≥3.8, PyTorch ≥2.0, PyMOR ≥2023.1, NumPy, tqdm, etc.).  

**Working Directory ($WORK):**  
The code expects an environment variable `$WORK` pointing to a directory where large datasets and results will be stored.  
Before running experiments, export `WORK` to a suitable path (if not set, it defaults to `~/palma_work`).  
All training data, model checkpoints, and outputs will be saved under `$WORK` in subfolders for each example.

**Utility Scripts:**  
For cluster users, the `utils/send_code.sh` and `utils/fetch_work.sh` scripts simplify running experiments remotely.  
Edit the SSH settings at the top of these scripts, then use `send_code.sh` to sync the repository to the remote server and `fetch_work.sh` to download the `$WORK/files` results back to your local `$WORK` directory.

---

## Model Descriptions

This repository implements several deep learning ROM architectures for parametric time-dependent PDEs.  
Each model approximates the high-fidelity solution \( u_h(t;\mu,\nu)\in \mathbb{R}^{N_h} \) (with \( \mu \) a geometric parameter, \( \nu \) a physical parameter, and \( t \) time) by a **low-dimensional** representation and reconstruction.

### POD-DL-ROM *(Fresca et al., 2020)*
Uses a Proper Orthogonal Decomposition matrix \( A\in\mathbb{R}^{N_h\times N_A} \) to reduce the full order dimension to \( N_A \ll N_h \).  
An autoencoder then compresses this POD space to a latent dimension \( n \) (Encoder: \( N_A \to n \), Decoder: \( n \to N_A \)), and a feed-forward network \( \phi_n \) maps \((\mu,\nu,t)\) to the latent coordinates.  
The resulting approximation is:

\[
\hat{u}_h(\mu,\nu,t) \approx A\,(\psi_{N_A}\circ \phi_n)(\mu,\nu,t)
\]

where \( \psi_{N_A} \) is the decoder.  
This is essentially the DL-ROM approach of Fresca & Manzoni, combining POD with a deep neural network latent model.

---

### DOD-DL-ROM
Extends the above by learning a **Deep Orthogonal Decomposition** (DOD) basis that varies with \(\mu\) and \(t\).  
A neural network \( \Phi_{\tilde V} \) produces a time- and parameter-dependent orthonormal basis \( V(\mu,t)\in\mathbb{R}^{N_A\times N'} \) (with \( N' \ll N_A \)).  
An autoencoder (Encoder: \( N' \to n \), Decoder: \( n \to N' \)) and a latent coefficient network \( \phi_n \) (input \((\mu,\nu,t)\), output \(\mathbb{R}^n\)) are trained in series.  
The full-order approximation is:

\[
\hat{u}_h(\mu,\nu,t) \approx V(\mu,t)\,\hat{q}_D(\mu,\nu,t),
\]

where \( \hat{q}_D = (\psi_{N'}\circ \phi_n)(\mu,\nu,t) \) are the decoded coefficients in the time-varying basis.  
This approach learns a **dynamic** reduced basis along with the parametric latent evolution, providing greater expressivity at the cost of a more complex training procedure.

---

### DOD+DFNN
A simpler variant of DOD-DL-ROM that **omits the autoencoder**.  
It uses the same learned DOD basis \( V(\mu,t) \) of dimension \( N' \), but replaces the autoencoder and latent model with a single deep feedforward network \( \Phi_{N'} \) that maps \((\mu,\nu,t)\) directly to an \( N' \)-dimensional coefficient vector.  
The approximation is:

\[
\hat{u}_h(\mu,\nu,t) \approx V(\mu,t)\,\Phi_{N'}(\mu,\nu,t)
\]

This model (akin to a baseline in the literature) is less expressive than DOD-DL-ROM but easier to train, since it relies on one network to predict coefficients in the adaptive basis (compare, for example, to the approach in Franco *et al.*, 2024).

---

### CoLoRA *(Berman & Peherstorfer, 2024)*
A hybrid approach that leverages a **stationary solution** as a reference.  
First, a *stationary* DOD basis \( V_{\text{stat}}(\mu) \) (time-independent) is learned for the steady-state of the system, and a corresponding coefficient network is trained for the stationary solution.  
Then a specialized CoLoRA network \( \Phi_\alpha \) (a deep feedforward architecture with custom low-rank layers) maps the physical parameters, time, and the pre-computed stationary solution into the solution space.  
In forward inference, this model produces the full state as:

\[
\hat{u}_h(\mu,\nu,t) \approx A \,\Phi_\alpha\!\big(\nu,\,t,\,\hat{u}^{\,\text{stat}}_{\mu,\nu}\big),
\]

where \( A \) is again a POD basis for the spatial field and \( \hat{u}^{\,\text{stat}}_{\mu,\nu} \) is the pre-reduced stationary solution.  
The CoLoRA-inspired model thus adjusts a low-rank stationary solution to predict the time-evolving solution.  
*(This architecture is only meaningful when the problem has a well-defined stationary/steady-state component.)*

---

## Training Pipeline

A typical workflow to train and evaluate the ROMs on a given example (say **Example 1: inviscid Burgers** problem) is as follows:

### 1. Data Generation
Solve the full-order model and generate training snapshots using PyMOR.  
Example for 400 parameter samples and 30 time steps:

```
python examples/ex01/data.py --Ns 400 --Nt 30
```

This produces compressed NumPy datasets in `examples/ex01/training_data/` (including full-order solutions, POD basis, normalization constants, etc.).

### 2. Model Training
Train the ROM networks on the generated dataset:

```
python examples/learning.py --example ex01_inviscid_burgers --epochs 1000 --restarts 5 --with-stationary
```

This trains each model (POD-DL-ROM, DOD+DFNN, DOD-DL-ROM, and CoLoRA if `--with-stationary` is set)  
and saves weights under `examples/ex01/state_dicts/`.

### 3. Testing and Evaluation

```
python examples/test.py --example ex01_inviscid_burgers
```

Evaluates each ROM type, producing error CSVs and metrics in `examples/ex01/benchmarks/`.

### 4. Analysis and Plots

```
python examples/analysis.py --example ex01_inviscid_burgers
```

Generates error plots, complexity charts, and singular value decay visualizations  
in `examples/ex01/benchmarks/analysis/`.

> **Note:** Replace `ex01_inviscid_burgers` with the desired example name (`ex02`, `ex03`, etc.).  
> Ensure `$WORK` is set to the desired storage path before running.

---

## Authors and Credit

This project was developed by **Dawid Kotowski** as part of his M.Sc. thesis under the supervision of **Prof. Dr. Mario Ohlberger** at the University of Münster.  

The code and methodologies build upon recent research in machine-learning ROMs – notably the DL-ROM frameworks by *Fresca and Manzoni* and the continuous low-rank adaptation ideas by *Berman and Peherstorfer*.  
Proper credit is due to these works, which inspired the models implemented here.
