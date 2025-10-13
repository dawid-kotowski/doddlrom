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

def plot_shared_params_vs_error(
    df: pd.DataFrame,
    example_name: str,
    outdir: Path,
    linestyles: tuple[str, ...] = ("-", "--", "-.", ":"),
    markers: tuple[str, ...] = ("o", "s", "^", "D", "P", "v", ">", "<", "h", "X"),
    show_r2: bool = True,
):
    """
    Scatter (points) + log-log fit (lines) per model with matching colors.
    y = C * x^slope  ⇔  log y = log C + slope * log x
    """
    outdir.mkdir(parents=True, exist_ok=True)
    colors = ["#4E79A7","#F28E2B","#E15759","#76B7B2",
                "#59A14F","#EDC948","#B07AA1","#FF9DA7","#9C755F","#BAB0AC"]

    models = [m for m in df["rom"].dropna().unique()]
    if not models:
        print("[analysis] No 'rom' values found; skipping plot.")
        return {}

    plt.figure()
    slopes: dict[str, float] = {}

    for i, model in enumerate(models):
        sub = df[df["rom"] == model]
        x = pd.to_numeric(sub["rel_L2G"], errors="coerce").to_numpy(dtype=float)
        y = pd.to_numeric(sub["params_nonzero"], errors="coerce").to_numpy(dtype=float)

        mask = (x > 0) & (y > 0) & np.isfinite(x) & np.isfinite(y)
        x, y = x[mask], y[mask]
        if x.size < 2:
            continue

        lx, ly = np.log(x), np.log(y)
        slope, intercept = np.polyfit(lx, ly, 1)
        slopes[model] = float(slope)
        yhat = intercept + slope * lx
        r2 = 1 - np.sum((ly - yhat) ** 2) / np.sum((ly - ly.mean()) ** 2) if show_r2 and lx.size > 1 else None

        xs = np.geomspace(x.min(), x.max(), 200)
        ys = np.exp(intercept) * xs ** slope

        c  = colors[i % len(colors)]
        ls = linestyles[i % len(linestyles)]
        mk = markers[i % len(markers)]

        plt.loglog(x, y, linestyle="none", marker=mk, markersize=6,
                   markerfacecolor=c, markeredgecolor=c, alpha=0.85, label=None)
        label = f"{model} fit slope={slope:.2f}" + (f", $R^2$={r2:.2f}" if r2 is not None else "")
        plt.loglog(xs, ys, linestyle=ls, color=c, linewidth=2.0, label=label)

    plt.xlabel(r"$\mathcal{E}_R$ (relative error)")
    plt.ylabel("# active weights")
    plt.title("Comparison: weights vs. error")
    plt.grid(True, which="both", linestyle=":", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "shared_params_vs_error.png", dpi=150)
    plt.close()
    return slopes

def plot_knw(
    example_name,
    outdir,
    zero_tol: float = 1e-14,
    head_frac: float = 0.02,
    tail_frac: float = 0.02,
    linestyles: tuple[str, ...] = ("-", "--"),
    markers: tuple[str, ...] = ("o", "s"),
    show_r2: bool = True,
):
    """
    Plot tail sums S(n) = sum_{k>n} sigma_k on semilogy (linear x, log y)
    with the same optical style as your shared_params_vs_error:
      - scatter points (colored markers)
      - same-colored fit line with custom linestyle
      - modern palette

    Fits S(n) ~ c * n^{-alpha} on a trimmed, non-near-zero window.
    """
    from pathlib import Path
    import numpy as np
    import matplotlib.pyplot as plt

    outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)
    path = Path(training_data_path(example_name) / f"pod_singular_values_{example_name}.npz")
    data = np.load(path)

    colors = ["#4E79A7","#F28E2B","#E15759","#76B7B2",
                "#59A14F","#EDC948","#B07AA1","#FF9DA7","#9C755F","#BAB0AC"]

    sigma_global = np.asarray(data['sigma_global_NA'], dtype=float).reshape(-1)
    sigma_sup = (np.asarray(data['sigma_mu_t_sup'], dtype=float).reshape(-1)
                 if 'sigma_mu_t_sup' in data.files
                 else np.max(np.asarray(data['sigma_mu_t'], dtype=float), axis=(0, 1)))

    sigma_global = np.maximum(sigma_global, 0.0)
    sigma_sup    = np.maximum(sigma_sup,    0.0)

    N_A = int(sigma_global.shape[0])
    n_vals = np.arange(1, N_A, dtype=int)

    def tail_sums(sig):
        tail = np.cumsum(sig[::-1])[::-1]
        S = np.zeros_like(n_vals, dtype=float)
        valid = n_vals + 1 < len(tail)
        S[valid] = tail[n_vals[valid] + 1]
        return S

    S_global = tail_sums(sigma_global)
    S_sup    = tail_sums(sigma_sup)

    def choose_fit_indices(S):
        mask = S > zero_tol
        if not np.any(mask):
            return np.array([], dtype=int)
        first = np.where(mask)[0][0]
        last  = np.where(mask)[0][-1]
        length = last - first + 1
        drop_h = int(max(0, round(head_frac * length)))
        drop_t = int(max(0, round(tail_frac * length)))
        lo, hi = first + drop_h, last - drop_t
        if lo > hi: lo, hi = first, last
        return np.arange(lo, hi + 1, dtype=int)

    def fit_power_law(n, S):
        idx = choose_fit_indices(S)
        if idx.size == 0:
            return None
        x = n[idx].astype(float)
        y = np.maximum(S[idx], zero_tol)
        lx, ly = np.log(x), np.log(y)
        slope, intercept = np.polyfit(lx, ly, 1)   # log S = a + b log n
        alpha = -slope
        c = np.exp(intercept)
        yhat = intercept + slope * lx
        R2 = 1 - np.sum((ly - yhat)**2)/np.sum((ly - ly.mean())**2) if show_r2 and lx.size > 1 else None
        return {"alpha": float(alpha), "c": float(c), "idx": idx, "R2": (None if R2 is None else float(R2))}

    fit_g = fit_power_law(n_vals, S_global)
    fit_s = fit_power_law(n_vals, S_sup)

    def draw_panel(xn, S, fit, title, fname, color, ls, mk, ylabel):
        plt.figure()
        mask_sc = S > 0
        plt.semilogy(xn[mask_sc], S[mask_sc],
                     linestyle="none", marker=mk, markersize=6,
                     markerfacecolor=color, markeredgecolor=color, alpha=0.85, label=None)

        if fit is not None:
            jj = fit["idx"]
            S_fit = fit["c"] * (xn[jj] ** (-fit["alpha"]))
            label = (rf"Fit: $S\approx c\,n^{{-\alpha}}$, "
                     rf"$\alpha\approx{fit['alpha']:.2f}$" +
                     (rf", $R^2={fit['R2']:.2f}$" if fit["R2"] is not None else ""))
            plt.semilogy(xn[jj], S_fit, linestyle=ls, color=color, linewidth=2.0, label=label)

        plt.xlabel('n'); plt.ylabel(ylabel)
        plt.title(title)
        plt.grid(True, which="both", linestyle=":", alpha=0.4)
        plt.legend()
        plt.tight_layout()
        plt.savefig(outdir / fname, dpi=150)
        plt.close()

    c_g, c_s = colors[0], colors[1]
    ls_g = linestyles[0 % len(linestyles)]
    ls_s = linestyles[1 % len(linestyles)]
    mk_g = markers[0 % len(markers)]
    mk_s = markers[1 % len(markers)]

    draw_panel(n_vals, S_global, fit_g,
               title='Global singular-values tail',
               fname="knw_global.png",
               color=c_g, ls=ls_g, mk=mk_g,
               ylabel='Tail sum $S(n)$')

    draw_panel(n_vals, S_sup, fit_s,
               title='Sup over $(\\mu,t)$ singular-values tail',
               fname="knw_sup.png",
               color=c_s, ls=ls_s, mk=mk_s,
               ylabel='$S_{\\sup}(n)$')


def plot_sample_error_curves(
    example_name: str,
    outdir: Path,
    max_lines: int = 6,
    y_top: float = 1.5,
    x_pad: float = 0.5,
    linestyles: tuple[str, ...] = ("-", "--", "-.", ":"),
    markers: tuple[str, ...] = ("o", "s", "^", "D", "P", "v", ">", "<", "h", "X")
):
    """
      - sample_error_pod.png :  E_S vs N_s, one curve per POD dimension N
      - sample_error_dod.png :  E_S vs N_s, one curve per DOD dimension N'
      Styling: matching scatter & line colors, cycled linestyles/markers, dashed ref slope.
    """
    outdir.mkdir(parents=True, exist_ok=True)

    colors = ["#4E79A7","#F28E2B","#E15759","#76B7B2",
                "#59A14F","#EDC948","#B07AA1","#FF9DA7","#9C755F","#BAB0AC"]

    def _fit_fixed_slope_powerlaw(Ns: np.ndarray, Es: np.ndarray, gamma: float):
        mask = (Ns > 0) & (Es > 0) & np.isfinite(Ns) & np.isfinite(Es)
        if np.count_nonzero(mask) < 3:
            return None
        x = np.log(Ns[mask]); y = np.log(Es[mask])
        logC = float(np.mean(y + gamma * x))
        return np.exp(logC)

    for model in ("pod", "dod"):
        df, path = load_error_decomp_df(example_name, model)
        if df is None:
            continue

        if model == "pod":
            dim_col  = "N"
            title    = r"POD: sample error $\mathcal{E}_S$ vs $N_s$"
            fname    = "sample_error_pod.png"
            xlabel   = r"$N_s$"
            ref_gamma = 1/4           # POD ~ N_s^{-1/4}
            ref_label = r"$N_s^{-1/4}$"
            dim_label = r"$N$"
        else:
            dim_col  = "N_prime"
            title    = r"DOD: sample error $\mathcal{E}_S$ vs $N_s$"
            fname    = "sample_error_dod.png"
            xlabel   = r"$N_s$"
            ref_gamma = 1/8           # DOD ~ N_s^{-1/8}
            ref_label = r"$N_s^{-1/8}$"
            dim_label = r"$N'$"

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

        for i, d in enumerate(dims):
            sub = agg[agg[dim_col] == d].sort_values("test_samples")
            if len(sub) == 0:
                continue
            c  = colors[i % len(colors)]
            ls = linestyles[i % len(linestyles)]
            mk = markers[i % len(markers)]
            xs = sub["test_samples"].to_numpy(dtype=float)
            ys = sub["E_S"].to_numpy(dtype=float)

            plt.plot(xs, ys, linestyle="none", marker=mk, markersize=6,
                     markerfacecolor=c, markeredgecolor=c, alpha=0.9, label=None)
            label = fr"{dim_label}={int(d)}" if float(d).is_integer() else fr"{dim_label}={d}"
            plt.plot(xs, ys, linestyle=ls, color=c, linewidth=2.0, label=label)

        Ns_stack, Es_stack = [], []
        for d in dims:
            sub = agg[agg[dim_col] == d]
            Ns_stack.append(sub["test_samples"].to_numpy(dtype=float))
            Es_stack.append(sub["E_S"].to_numpy(dtype=float))
        if Ns_stack:
            Ns_stack = np.concatenate(Ns_stack)
            Es_stack = np.concatenate(Es_stack)
            C = _fit_fixed_slope_powerlaw(Ns_stack, Es_stack, ref_gamma)
            if C is not None and len(all_ns):
                xs_ref = np.array(sorted(set(all_ns)), dtype=float)
                y_ref  = C * (xs_ref ** (-ref_gamma))
                plt.plot(xs_ref, y_ref, linestyle="--", color="black", linewidth=1.8, label=ref_label)
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
            if ymax > y_top:
                print(f"[analysis] Warning: {model.upper()} sample-error max {ymax:.2e} exceeds y_top={y_top}; plotting clipped.")
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
