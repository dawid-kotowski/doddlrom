# DOD-DL-ROM

Deep-learning reduced-order models for parametric, time-dependent PDEs with dynamic orthogonal decompositions.

## What This Repository Contains

This repository provides:

- Data generation pipelines (`examples/ex0*/data.py`) for four PDE settings.
- Training pipelines for several ROM architectures (`examples/learning.py`, `examples/test.py`).
- Evaluation scripts (`examples/ex0*/eval.py`) and post-processing (`examples/analysis.py`).
- Core model implementations in [`core/reduced_order_models.py`](core/reduced_order_models.py).

## Repository Structure

```text
doddlrom/
├── core/
│   ├── reduced_order_models.py      # ROM architectures + trainers + forward wrappers
│   ├── configs/parameters.py         # Example-specific dimensions/hyperparameters
│   ├── analytic/                     # projection, decomposition, error utilities
│   └── bindings/                     # DUNE/Python bridge used by ex04
├── examples/
│   ├── ex01/                         # 2D advection-diffusion-reaction (stationary+instationary)
│   ├── ex02/                         # 2D nonlinear advection-diffusion with parametric source
│   ├── ex03/                         # 1D heat equation with parametric step initial condition
│   ├── ex04/                         # DUNE DarcyFlow transport example (bindings required)
│   ├── learning.py                   # train baseline ROMs for one example
│   ├── test.py                       # profile/sweep benchmarking across ROMs
│   └── analysis.py                   # plots from benchmark CSV outputs
├── utils/
│   ├── paths.py                      # WORK-based output locations
│   └── visualizer.py                 # plotting helpers used by eval scripts
├── requirements.lock.txt
└── setup.py
```

## Installation

### 1. Python environment

Recommended: Python `3.10+`.

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .
```

This installs direct runtime dependencies used by the project scripts (`torch`, `pymor`, `numpy`, `scipy`, `matplotlib`, `pandas`, `meshio`, `tqdm`).

### 2. Set runtime paths

The project writes/reads data from `$WORK`. Always set it explicitly:

```bash
export WORK=/absolute/path/to/your/workdir
```

Expected layout under `$WORK`:

- `$WORK/training_data/<example>`
- `$WORK/state_dicts/<example>`
- `$WORK/benchmarks/<example>`

For headless servers, also set:

```bash
export MPLCONFIGDIR=/tmp/mpl
```

### 3. System tools

- `ex01`/`ex01 eval` use Gmsh via pyMOR meshing. Install `gmsh` system-wide.
- `ex04` requires additional DUNE-based bindings (see below).

## ex04: DUNE DarcyFlow Bindings (Required Only for ex04 Data/Eval)

`ex04` imports `dune.*` and `dune.darcyflow._darcyflow`. If this binding is missing, `ex04/eval.py` and `ex04/data.py` will fail with `ModuleNotFoundError: No module named 'dune'`.

The bindings come from:

- <https://github.com/dawid-kotowski/darcyflow>

From that repository README:

- `cmake >= 3.16` is required.
- Build is driven by `dunecontrol all` (optionally with `--opts=<opts-file>` for compiler/CMake flags).

Minimal flow:

```bash
git clone https://github.com/dawid-kotowski/darcyflow.git
cd darcyflow
# configure your DUNE opts if needed
# then build
dunecontrol all
```

After build, activate the generated Python-path environment from that build (or manually export the build Python paths), then verify:

```bash
python -c "from dune.darcyflow import _darcyflow; print('dune-darcyflow OK')"
```

## ROM Formulation (Aligned with `core/reduced_order_models.py`)

Let the high-fidelity state be `u_h(t; mu, nu) in R^{N_h}`.

The code stores POD ambient bases:

- `A in R^{N_h x N_A}` for DOD-based models.
- `A_P in R^{N_h x N}` for POD-based models.

A Dirichlet/initial shift `u_{t,0}` is loaded from `dirichlet_shift_<example>.npz`, and predictions are lifted back to full order as:

- `u_hat = u_{t,0} + A * y` (DOD path), or
- `u_hat = u_{t,0} + A_P * y` (POD path).

Implemented ROMs:

1. **innerDOD**
   - Learns a dynamic basis `V(mu, t) in R^{N_A x N'}` with orthonormalized columns (`Orth` block).

2. **DOD+DFNN**
   - Coefficient net `phi_{N'}(mu, nu, t)` predicts `N'` coefficients directly.
   - Reduced state: `y = V(mu,t) * phi_{N'}(mu,nu,t)`.

3. **DOD-DL-ROM**
   - Coefficient net predicts latent `n` vector, decoder maps `n -> N'`.
   - Reduced state: `y = V(mu,t) * decoder(phi_n(mu,nu,t))`.

4. **POD-DL-ROM**
   - Coefficient net predicts latent `n`, decoder maps `n -> N` in fixed POD space.
   - Reduced state: `y = decoder(phi_n(mu,nu,t))`.

5. **CoLoRA** (only meaningful for examples with stationary data)
   - Uses stationary reduced representation plus layered low-rank updates (`CoLoRA` module).
   - In this repository workflow, this is available for examples where reduced stationary data is generated (notably `ex01`, and `ex03` if generated).

## Mathematical Example Summary

### ex01 (2D advection-diffusion-reaction on polygon with holes)

- Domain: non-convex polygon with two circular holes.
- Parameters: `mu` (advection angle), `nu` (diffusion weight).
- Instationary equation (conceptually):
  - `u_t - div(kappa(x,nu) grad u) + b(mu) · grad u + c u = 0`
  - with mixed boundary treatment and Dirichlet shift handling.

### ex02 (2D nonlinear advection-diffusion with parametric source)

- Domain: unit rectangle.
- Parameters: `mu=(c_x,c_y,theta_0)` and scalar `nu`.
- Time-varying advection direction from `theta(t)=theta_0 + 0.25 sin(2 pi t)`.
- RHS: weighted Gaussian mixture with `nu`-dependent amplitudes.

### ex03 (1D heat equation with parametric initial front)

- Domain: interval `[0,1]`, homogeneous Dirichlet boundaries.
- Parameters: front center `mu`, diffusivity `nu`.
- PDE: `u_t - nu u_xx = 0`.
- Initial condition: smoothed step `0.5*(1+tanh((x-mu)/eps))`.

### ex04 (DUNE DarcyFlow transport model)

- Full-order model comes from the external `dune-darcyflow` binding.
- The local bridge maps parameter blocks as `mu` and `nu` vectors in the DUNE solver interface.
- Uses `l2_product` Gram matrix (not `h1_0_semi_product`).

## Training and Evaluation Pipeline

Run all commands from the repository root (`doddlrom/`) with the virtualenv active.

### A. Generate data

```bash
python examples/ex01/data.py
python examples/ex02/data.py
python examples/ex03/data.py
python examples/ex04/data.py   # requires dune-darcyflow bindings
```

Useful overrides:

- `--Ns` number of parameter samples
- `--Nt` number of time steps
- `--N` POD reduction size (where applicable)

### B. Train baseline models (`learning.py`)

```bash
python examples/learning.py --example ex01 --epochs 200 --restarts 3
python examples/learning.py --example ex02 --epochs 200 --restarts 1
python examples/learning.py --example ex03 --epochs 200 --restarts 1
python examples/learning.py --example ex04 --epochs 200 --restarts 5
```

Notes:

- `--epochs` and `--restarts` now apply to both innerDOD and downstream ROM stages.
- Add `--with-stationary` only when stationary reduced data exists (e.g., `ex01`; `ex03` after generating stationary datasets).

### C. Evaluate trained models (`eval.py`)

```bash
python examples/ex01/eval.py
python examples/ex02/eval.py
python examples/ex03/eval.py
python examples/ex04/eval.py   # requires dune-darcyflow bindings
```

- Scripts load weights from `$WORK/state_dicts/<example>`.
- If Qt is unavailable, visualization is skipped and metric/csv output continues.

### D. Run ROM profile sweeps (`test.py`)

```bash
python examples/test.py --example ex01
python examples/test.py --example ex02
python examples/test.py --example ex03
python examples/test.py --example ex04
```

This appends to `$WORK/benchmarks/<example>/rom_sweep.csv`.

### E. Generate analysis plots

```bash
python examples/analysis.py --example ex01
python examples/analysis.py --example ex02
python examples/analysis.py --example ex03
python examples/analysis.py --example ex04
```

## Practical Notes

- `analysis.py` requires `rom_sweep.csv` and error decomposition CSVs; run `test.py` and `eval.py` first.
- `ex03/eval.py` requires ex03 training data and weights; if they are not in `$WORK`, it will fail with missing file errors.
- `ex04/learning.py` can run from precomputed reduced data without DUNE bindings, but `ex04/data.py` and `ex04/eval.py` require the bindings.

## Authors and Credit

- **Dawid Kotowski** (project author, MSc thesis context at University of Münster).
- Supervision context: **Prof. Dr. Mario Ohlberger**.
- Model lineage and inspiration include DL-ROM/POD-DL-ROM and continuous low-rank adaptation literature.
- `ex04` full-order backend credit: DUNE ecosystem and `darcyflow` binding repository.

