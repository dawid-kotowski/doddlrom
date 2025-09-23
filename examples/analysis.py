'''
Saving Error Analysis and KnW plots to examples/{example_name}/benchmarks/analysis
for each {example_name}

Run programm
-------------------------
python examples/analysis.py
--example "ex{number}"
-------------------------

'''
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def load_rom_sweep(example_name):
    path = Path(f"examples/{example_name}/benchmarks/rom_sweep.csv")
    return pd.read_csv(path)

def plot_params_vs_error(df, example_name, outdir):
    models = df['rom'].unique()
    for model in models:
        sub = df[df['rom'] == model]
        x = sub['rel_L2G'].values.astype(float)
        y = sub['params_nonzero'].values.astype(float)

        # regression in log-log space
        coeffs = np.polyfit(np.log(x), np.log(y), 1)
        slope, intercept = coeffs
        fit_line = np.exp(intercept) * x**slope

        plt.figure()
        plt.loglog(x, y, 'o', label=model)
        plt.loglog(x, fit_line, '-', label=f"Fit slope={slope:.2f}")
        plt.xlabel(r"$\mathcal{E}_R$ (relative error)")
        plt.ylabel("# active weights")
        plt.title(f"{model}: weights vs. error")
        plt.legend()
        plt.tight_layout()
        plt.savefig(outdir / f"{model}_params_vs_error.png")
        plt.close()

def plot_shared_params_vs_error(df, example_name, outdir):
    models = df['rom'].unique()
    plt.figure()
    for model in models:
        sub = df[df['rom'] == model]
        x = sub['rel_L2G'].values.astype(float)
        y = sub['params_nonzero'].values.astype(float)

        coeffs = np.polyfit(np.log(x), np.log(y), 1)
        slope, intercept = coeffs
        fit_line = np.exp(intercept) * x**slope

        plt.loglog(x, y, 'o', label=f"{model}")
        plt.loglog(x, fit_line, '-', label=f"{model} fit slope={slope:.2f}")

    plt.xlabel(r"$\mathcal{E}_R$ (relative error)")
    plt.ylabel("# active weights")
    plt.title("Comparison: weights vs. error")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "shared_params_vs_error.png")
    plt.close()

def plot_error_vs_samples(df, example_name, outdir):
    models = df['rom'].unique()
    plt.figure()
    for model in models:
        sub = df[df['rom'] == model]
        x = sub['N_s'].values.astype(float)
        y = sub['rel_L2G'].values.astype(float)
        plt.loglog(x, y, 'o-', label=model)
    plt.xlabel(r"$N_s$ (samples)")
    plt.ylabel(r"$\mathcal{E}_R$ (relative error)")
    plt.title("Error vs. number of samples")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "error_vs_samples.png")
    plt.close()

def plot_knw(example_name, outdir):
    # ---- load ----
    path = Path(f"examples/{example_name}/training_data/pod_singular_values_{example_name}.npz")
    data = np.load(path)

    sigma_global = data['sigma_global_NA'].astype(float)  # [N_A]
    sigma_sup = (data['sigma_mu_t_sup'].astype(float)
                 if 'sigma_mu_t_sup' in data
                 else np.max(data['sigma_mu_t'].astype(float), axis=(0, 1)))

    sigma_global = np.maximum(sigma_global, 1e-16)
    sigma_sup    = np.maximum(sigma_sup,    1e-16)

    N_A = int(sigma_global.shape[0])
    n_max = max(10, min(100, N_A - 1))
    n_vals = np.arange(1, n_max, dtype=int)

    # tail sums S(n) = sum_{k>n} sigma_k
    S_global = np.array([np.sum(sigma_global[n+1:]) for n in n_vals])
    S_sup    = np.array([np.sum(sigma_sup[n+1:])    for n in n_vals])

    drop_head = max(2, int(0.02 * len(n_vals)))
    drop_tail = max(2, int(0.02 * len(n_vals)))
    fit_slice = slice(drop_head, len(n_vals) - drop_tail)

    x_fit = n_vals[fit_slice].astype(float)
    y_g_fit = np.maximum(S_global[fit_slice], 1e-16)
    y_s_fit = np.maximum(S_sup[fit_slice],    1e-16)

    # log–log linear regression: log S = a + b log n, with alpha = -b
    slope_g, intercept_g = np.polyfit(np.log(x_fit), np.log(y_g_fit), 1)
    slope_s, intercept_s = np.polyfit(np.log(x_fit), np.log(y_s_fit), 1)
    alpha = -slope_g
    beta  = -slope_s

    Sg_fit_line = np.exp(intercept_g) * n_vals**slope_g
    Ss_fit_line = np.exp(intercept_s) * n_vals**slope_s

    # ---- plots ----
    # Global
    plt.figure()
    plt.loglog(n_vals, S_global, label='Global tail sum S(n)')
    plt.loglog(n_vals, Sg_fit_line, linestyle='--',
               label=f'Fit: n^({slope_g:.2f})  →  α≈{alpha:.2f}')
    plt.xlabel('n'); plt.ylabel('Tail sum S(n)')
    plt.title('Global singular-values tail (log–log)')
    plt.legend(); plt.tight_layout()
    plt.savefig(outdir / "knw_global.png")
    plt.close()

    # Supremum
    plt.figure()
    plt.loglog(n_vals, S_sup, label='Sup tail sum S_sup(n)')
    plt.loglog(n_vals, Ss_fit_line, linestyle='--',
               label=f'Fit: n^({slope_s:.2f})  →  β≈{beta:.2f}')
    plt.xlabel('n'); plt.ylabel('Tail sum S_sup(n)')
    plt.title('Sup over (μ,t) singular-values tail (log–log)')
    plt.legend(); plt.tight_layout()
    plt.savefig(outdir / "knw_sup.png")
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--example", type=str, required=True)
    args = parser.parse_args()
    example_name = args.example

    outdir = Path(f"examples/{example_name}/benchmarks/analysis")
    outdir.mkdir(parents=True, exist_ok=True)

    df = load_rom_sweep(example_name)

    plot_params_vs_error(df, example_name, outdir)
    plot_shared_params_vs_error(df, example_name, outdir)
    plot_error_vs_samples(df, example_name, outdir)
    plot_knw(example_name, outdir)

if __name__ == "__main__":
    main()
