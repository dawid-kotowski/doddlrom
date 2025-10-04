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
from utils.paths import benchmarks_path, training_data_path
import matplotlib.pyplot as plt
from pathlib import Path

def load_rom_sweep(example_name):
    path = Path( benchmarks_path(example_name) / f"rom_sweep.csv")
    return pd.read_csv(path)

def load_error_decomp_df(example_name: str, model: str):
    csv_path = _find_error_csv(example_name, model)
    if csv_path is None:
        print(f"[analysis] No CSV found for {model.upper()} at benchmarks/{example_name}/...")
        return None, None
    try:
        df = pd.read_csv(csv_path)
        return df, csv_path
    except Exception as e:
        print(f"[analysis] Failed reading {csv_path.name}: {e}")
        return None, None

def _find_error_csv(example_name: str, model: str):
    """
    Guard for CSV files.
    """
    bench = Path(benchmarks_path(example_name))
    candidates = [
        bench / f"error_decomposiiton_{model}.csv",   # legacy typo
        bench / f"error_decomp_{model}.csv",
        bench / f"error_decomposition_{model}.csv",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None

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

def plot_knw(example_name, outdir, zero_tol=1e-14, head_frac=0.02, tail_frac=0.02):
    """
    Plot tail sums S(n) = sum_{k>n} sigma_k on a semilogy (linear x, log y).
    Fit S(n) ~ c * n^{-alpha} using log–log regression, but *only* over
    the non-near-zero region of S to avoid KNW artifacts once tails vanish.

    Parameters
    ----------
    zero_tol : float
        Entries of S(n) <= zero_tol are considered "near zero" and excluded from the fit window.
    head_frac, tail_frac : float in [0, 0.5)
        Optional trimming (fractions of the valid fit window) to avoid edge effects.
    """
    outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)
    path = Path(training_data_path(example_name) / f"pod_singular_values_{example_name}.npz")
    data = np.load(path)

    sigma_global = np.asarray(data['sigma_global_NA'], dtype=float).reshape(-1)
    sigma_sup = (np.asarray(data['sigma_mu_t_sup'], dtype=float).reshape(-1)
                 if 'sigma_mu_t_sup' in data.files
                 else np.max(np.asarray(data['sigma_mu_t'], dtype=float), axis=(0, 1)))

    sigma_global = np.maximum(sigma_global, 0.0)
    sigma_sup    = np.maximum(sigma_sup,    0.0)

    N_A = int(sigma_global.shape[0])
    n_vals = np.arange(1, N_A, dtype=int)

    # Tail sums via reverse cumsum: tail[k] = sum_{j>=k} sigma_j
    def tail_sums(sig):
        tail = np.cumsum(sig[::-1])[::-1]
        S = np.zeros_like(n_vals, dtype=float)
        valid = n_vals + 1 < len(tail)
        S[valid] = tail[n_vals[valid] + 1]
        return S

    S_global = tail_sums(sigma_global)
    S_sup    = tail_sums(sigma_sup)

    # Helper: choose fit indices only where S > zero_tol, then trim head/tail a bit
    def choose_fit_indices(S):
        mask = S > zero_tol
        if not np.any(mask):
            return np.array([], dtype=int)
        last = np.where(mask)[0][-1]
        first = np.where(mask)[0][0]
        
        length = last - first + 1
        drop_h = int(max(0, round(head_frac * length)))
        drop_t = int(max(0, round(tail_frac * length)))
        lo = first + drop_h
        hi = last - drop_t
        if lo > hi:
            lo, hi = first, last
        return np.arange(lo, hi + 1, dtype=int)

    # Fit S ~ c * n^{-alpha} via log–log regression, restricted to chosen indices
    def fit_power_law(n, S):
        idx = choose_fit_indices(S)
        if idx.size == 0:
            return None
        x = n[idx].astype(float)
        y = np.maximum(S[idx], zero_tol)
        slope, intercept = np.polyfit(np.log(x), np.log(y), 1)  # log S = a + b log n
        alpha = -slope
        c = np.exp(intercept)
        return {"alpha": float(alpha), "c": float(c), "idx": idx}

    fit_g = fit_power_law(n_vals, S_global)
    fit_s = fit_power_law(n_vals, S_sup)

    # Build fitted lines only on their respective fit windows
    Sg_fit = np.full_like(S_global, np.nan, dtype=float)
    if fit_g is not None:
        jj = fit_g["idx"]
        Sg_fit[jj] = fit_g["c"] * (n_vals[jj] ** (-fit_g["alpha"]))

    Ss_fit = np.full_like(S_sup, np.nan, dtype=float)
    if fit_s is not None:
        jj = fit_s["idx"]
        Ss_fit[jj] = fit_s["c"] * (n_vals[jj] ** (-fit_s["alpha"]))

    # ---------- Plots (linear x, log y) ----------
    # Global
    plt.figure()
    plt.semilogy(n_vals, np.maximum(S_global, zero_tol), label='Global tail sum $S(n)$')
    if fit_g is not None:
        plt.semilogy(n_vals, np.maximum(Sg_fit, zero_tol), linestyle='--',
                     label=f'Fit on non-zero region: $S\\approx c n^{{-\\alpha}}$, '
                           f'$\\alpha\\approx{fit_g["alpha"]:.2f}$')
    plt.xlabel('n'); plt.ylabel('Tail sum $S(n)$')
    plt.title('Global singular-values tail')
    plt.legend(); plt.tight_layout()
    plt.savefig(outdir / "knw_global.png", dpi=150)
    plt.close()

    # Supremum
    plt.figure()
    plt.semilogy(n_vals, np.maximum(S_sup, zero_tol), label='Sup tail sum $S_{\\sup}(n)$')
    if fit_s is not None:
        plt.semilogy(n_vals, np.maximum(Ss_fit, zero_tol), linestyle='--',
                     label=f'Fit on non-zero region: $S\\approx c n^{{-\\beta}}$, '
                           f'$\\beta\\approx{fit_s["alpha"]:.2f}$')
    plt.xlabel('n'); plt.ylabel('$S_{\\sup}(n)$')
    plt.title('Sup over $(\\mu,t)$ singular-values tail')
    plt.legend(); plt.tight_layout()
    plt.savefig(outdir / "knw_sup.png", dpi=150)
    plt.close()

def plot_sample_error_curves(
    example_name: str,
    outdir: Path,
    max_lines: int = 6,
    y_top: float = 1.5,
    x_pad: float = 0.5
):
    """
      - sample_error_pod.png   : E_S vs N     (one line per test_samples)
      - sample_error_dod.png   : E_S vs N'    (one line per test_samples)

    Linear x-axis, integer ticks at the dimensions present.
    """
    outdir.mkdir(parents=True, exist_ok=True)

    def _fit_fixed_slope_powerlaw(Ns: np.ndarray, Es: np.ndarray, gamma: float):
        mask = (Ns > 0) & (Es > 0)
        if np.count_nonzero(mask) < 3:
            return None
        x = np.log(Ns[mask])
        y = np.log(Es[mask])
        logC = float(np.mean(y + gamma * x))
        C = np.exp(logC)
        return C

    for model in ("pod", "dod"):
        df, path = load_error_decomp_df(example_name, model)
        if df is None:
            continue

        if model == "pod":
            dim_col = "N"
            title   = r"POD: sample error $\mathcal{E}_S$ vs $N_s$"
            fname   = "sample_error_pod.png"
            xlabel  = r"$N_s$"
            ref_gamma = 1/4   # POD ~ N_s^{-1/4}
            ref_label = r"$N_s^{-1/4}$"
        else:
            dim_col = "N_prime"
            title   = r"DOD: sample error $\mathcal{E}_S$ vs $N_s$"
            fname   = "sample_error_dod.png"
            xlabel  = r"$N_s$"
            ref_gamma = 1/8   # DOD ~ N_s^{-1/8}
            ref_label = r"$N_{s_2}^{-1/4}$"

        df = df.copy()
        df[dim_col] = pd.to_numeric(df[dim_col], errors="coerce")
        df["test_samples"] = pd.to_numeric(df["test_samples"], errors="coerce")
        df["E_S"] = pd.to_numeric(df["E_S"], errors="coerce")
        df = df.dropna(subset=[dim_col, "test_samples", "E_S"])

        agg = (
            df.groupby([dim_col, "test_samples"], as_index=False, dropna=False)
              .agg(E_S=("E_S", "mean"))
        )

        dims = sorted(agg[dim_col].unique())

        if max_lines is not None:
            dims = dims[:max_lines]

        all_ns = sorted(agg["test_samples"].unique())
        plt.figure()

        for d in dims:
            sub = agg[agg[dim_col] == d].sort_values("test_samples")
            if len(sub) == 0:
                continue
            plt.plot(sub["test_samples"].values,
                     sub["E_S"].values,
                     marker="o", linewidth=2, label=fr"${dim_col}={int(d)}$" if float(d).is_integer() else fr"${dim_col}={d}$")

        Ns_stack, Es_stack = [], []
        for d in dims:
            sub = agg[agg[dim_col] == d]
            Ns_stack.append(sub["test_samples"].to_numpy(dtype=float))
            Es_stack.append(sub["E_S"].to_numpy(dtype=float))
        if len(Ns_stack):
            Ns_stack = np.concatenate(Ns_stack)
            Es_stack = np.concatenate(Es_stack)
            C = _fit_fixed_slope_powerlaw(Ns_stack, Es_stack, ref_gamma)
            if C is not None:
                xs = np.array(sorted(set(all_ns)), dtype=float)
                y_ref = C * (xs ** (-ref_gamma))
                plt.plot(xs, y_ref, "--", color="black", linewidth=1.8, label=ref_label)
            else:
                print(f"[analysis] Not enough positive points to fit {ref_label} for {model.upper()}.")

        if len(all_ns) > 0:
            xmin, xmax = min(all_ns), max(all_ns)
            plt.xlim(xmin - x_pad, xmax + x_pad)
            if len(all_ns) <= 15 and np.allclose(all_ns, np.round(all_ns)):
                plt.xticks(all_ns)

        plt.xlabel(xlabel)
        plt.ylabel(r"$\mathcal{E}_S$")
        plt.title(title)
        plt.grid(True, which="both", linestyle=":", alpha=0.4)
        plt.legend()

        if y_top is not None:
            ymax = 0.0
            for line in plt.gca().get_lines():
                ydata = np.asarray(line.get_ydata(), dtype=float)
                if ydata.size:
                    ymax = max(ymax, float(np.nanmax(ydata)))
            plt.ylim(0.0, y_top)

        plt.tight_layout()
        outpath = outdir / fname
        plt.savefig(outpath, dpi=150)
        plt.close()

def _nearest_test_samples(df: pd.DataFrame, target: int) -> int:
    """Pick the available 'test_samples' value closest to target."""
    vals = sorted(df["test_samples"].unique())
    if len(vals) == 0:
        return target
    return min(vals, key=lambda v: abs(v - target))

def plot_error_decomposition_figures(
    example_name: str,
    outdir: Path,
    Ns_target: int = 100,
    y_top: float = 10.0
):
    """
      - error_decomp_pod.png : POD E_R (red), UB (black dashed), LB (black dotted),
                               stacked fills for E_NN (blue), E_S (green), E_POD (pink) vs N (log x by default).
      - error_decomp_dod.png : same, with DOD terms and N' on x.
    """
    outdir.mkdir(parents=True, exist_ok=True)

    for model in ("pod", "dod"):
        df, path = load_error_decomp_df(example_name, model)
        if df is None:
            continue

        if model == "pod":
            xcol, comp_P, comp_Pinf, xlabel, fname = "N", "E_POD", "E_POD_inf", r"$N$", "error_decomp_pod.png"
            P_label = r"$\mathcal{E}_{\text{POD}}$"
            Pinfty_label = r"$\mathcal{E}_{\text{POD},\infty}$"
            UB_label = r"$\mathcal{E}_{\mathrm{NN}}+\mathcal{E}_S+\mathcal{E}_{\text{POD}}$"
        else:
            xcol, comp_P, comp_Pinf, xlabel, fname = "N_prime", "E_DOD", "E_DOD_inf", r"$N'$", "error_decomp_dod.png"
            P_label = r"$\mathcal{E}'_{\text{DOD}}$"
            Pinfty_label = r"$\mathcal{E}'_{\text{DOD},\infty}$"
            UB_label = r"$\mathcal{E}_{\mathrm{NN}}+\mathcal{E}_S+\mathcal{E}'_{\text{DOD}}$"

        use_Ns = _nearest_test_samples(df, Ns_target)
        sub = df[df["test_samples"] == use_Ns].copy()
        if len(sub) == 0:
            print(f"[analysis] No rows for {model.upper()} with test_samples≈{Ns_target}. Skipping.")
            continue

        cols = [xcol, "E_R", "E_S", comp_P, comp_Pinf, "E_NN", "upper_bound", "lower_bound", "m", "M"]
        present = [c for c in cols if c in sub.columns]
        sub = (sub[present]
               .groupby(xcol, as_index=False)
               .mean(numeric_only=True)
               .sort_values(xcol))

        x   = sub[xcol].astype(float).values
        ER  = np.nan_to_num(sub["E_R"].astype(float).values, nan=0.0)
        ES  = np.nan_to_num(sub["E_S"].astype(float).values, nan=0.0)
        EP  = np.nan_to_num(sub[comp_P].astype(float).values, nan=0.0)
        ENN = np.nan_to_num(sub["E_NN"].astype(float).values, nan=0.0)
        EPinfty = np.nan_to_num(sub[comp_Pinf].astype(float).values, nan=0.0)

        if "upper_bound" in sub.columns:
            UB = np.nan_to_num(sub["upper_bound"].astype(float).values, nan=0.0)
        else:
            UB = ENN + ES + EP
        if "lower_bound" in sub.columns:
            LB = np.nan_to_num(sub["lower_bound"].astype(float).values, nan=0.0)
        else:
            m = sub["m"].astype(float).values if "m" in sub.columns else np.ones_like(x)
            M = sub["M"].astype(float).values if "M" in sub.columns else np.ones_like(x)
            LB = (m / np.maximum(M, 1e-12)) * EPinfty

        # --- stacked fills ---
        zeros = np.zeros_like(x)
        band1_top = ENN
        band2_top = ENN + ES
        band3_top = ENN + ES + EP   # this equals UB if UB=ENN+ES+EP

        plt.figure()

        # bottom band: E_NN
        plt.fill_between(x, zeros, band1_top, facecolor="#8fb3ff", alpha=0.35, label=r"$\mathcal{E}_{\mathrm{NN}}$")
        # middle band: E_S
        plt.fill_between(x, band1_top, band2_top, facecolor="#a8e6a1", alpha=0.35, label=r"$\mathcal{E}_S$")
        # top band: E_POD or E_DOD
        plt.fill_between(x, band2_top, band3_top, facecolor="#f4a3b4", alpha=0.35, label=P_label)

        # Lines
        plt.plot(x, ER, color="red", linewidth=2.0, label=r"$\mathcal{E}_R$")
        plt.plot(x, UB, color="black", linestyle="--", linewidth=2.0, label=UB_label)
        plt.plot(x, LB, color="black", linestyle=":", linewidth=2.0, label=rf"$\frac{{m}}{{M}}\,${Pinfty_label}")

        # scales / labels
        plt.xscale("linear")
        plt.xlabel(xlabel)
        plt.ylabel("error")
        plt.title(f"{model.upper()} error decomposition (N_s ≈ {use_Ns})")
        plt.grid(True, which="both", linestyle=":", alpha=0.4)
        plt.legend(loc="best")

        # uniform y-limit ~10
        ymax_data = np.nanmax([band3_top.max(), UB.max(), ER.max(), LB.max()])
        if ymax_data > y_top:
            print(f"[analysis] Warning: {model.upper()} max {ymax_data:.2f} exceeds y_top={y_top}; plotting clipped.")
        plt.ylim(0.0, y_top)

        plt.tight_layout()
        outpath = outdir / fname
        plt.savefig(outpath, dpi=150)
        plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--example", type=str, required=True)
    args = parser.parse_args()
    example_name = args.example

    outdir = Path(benchmarks_path(example_name) / f"analysis")
    outdir.mkdir(parents=True, exist_ok=True)

    df = load_rom_sweep(example_name)

    plot_params_vs_error(df, example_name, outdir)
    plot_shared_params_vs_error(df, example_name, outdir)
    plot_knw(example_name, outdir)
    plot_sample_error_curves(example_name, outdir)
    plot_error_decomposition_figures(example_name, outdir)

if __name__ == "__main__":
    main()
