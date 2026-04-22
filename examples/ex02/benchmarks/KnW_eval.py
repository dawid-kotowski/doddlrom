import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import AutoMinorLocator, LogLocator, NullFormatter
from utils.paths import training_data_path

example_name = 'ex02'

path = training_data_path(example_name) / f'pod_singular_values_{example_name}.npz'
data = np.load(path)

sigma_global = data['sigma_global_NA'].astype(float)
sigma_sup = data['sigma_mu_t_sup'].astype(float) if 'sigma_mu_t_sup' in data else np.max(data['sigma_mu_t'].astype(float), axis=(0, 1))

eps = 1e-16
fit_cutoff = 1e-8
n_plot_max = 50

sigma_global = np.maximum(sigma_global, eps)
sigma_sup = np.maximum(sigma_sup, eps)

n_max_available = max(1, min(sigma_global.shape[0], sigma_sup.shape[0]) - 2)
n_max = min(n_plot_max, n_max_available)
n_vals = np.arange(1, n_max + 1, dtype=int)

S_global = np.array([np.sum(sigma_global[n + 1:]) for n in n_vals], dtype=float)
S_sup = np.array([np.sum(sigma_sup[n + 1:]) for n in n_vals], dtype=float)


def fit_loglog(x, y, cutoff):
    mask = y > cutoff
    if np.count_nonzero(mask) < 2:
        mask = y > 0.0
    if np.count_nonzero(mask) < 2:
        return np.nan, np.nan
    return np.polyfit(np.log(x[mask]), np.log(y[mask]), 1)


def preferred_serif_fonts():
    available_fonts = {font.name for font in font_manager.fontManager.ttflist}
    candidates = [
        'Latin Modern Roman',
        'LM Roman 10',
        'CMU Serif',
        'Computer Modern Roman',
        'STIX Two Text',
        'STIXGeneral',
        'DejaVu Serif',
    ]
    selected = [font_name for font_name in candidates if font_name in available_fonts]
    return selected if selected else ['serif']


x_fit = n_vals.astype(float)
slope_g, intercept_g = fit_loglog(x_fit, S_global, fit_cutoff)
slope_s, intercept_s = fit_loglog(x_fit, S_sup, fit_cutoff)

alpha = -slope_g if np.isfinite(slope_g) else np.nan
beta = -slope_s if np.isfinite(slope_s) else np.nan

print(f"Estimated alpha (global tail ~ n^(-alpha)): {alpha:.3f}")
print(f"Estimated beta  (sup tail ~ n^(-beta))   : {beta:.3f}")

S_global_plot = np.maximum(S_global, eps)
S_sup_plot = np.maximum(S_sup, eps)

plot_rc = {
    'font.family': 'serif',
    'font.serif': preferred_serif_fonts(),
    'mathtext.fontset': 'cm',
    'axes.unicode_minus': False,
    'font.size': 14,
    'axes.labelsize': 17,
    'xtick.labelsize': 15,
    'ytick.labelsize': 15,
    'legend.fontsize': 18,
}

with plt.rc_context(plot_rc):
    fig, ax = plt.subplots(figsize=(7.0, 4.4), constrained_layout=True)
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    for spine in ('top', 'right'):
        ax.spines[spine].set_visible(False)
    for spine in ('left', 'bottom'):
        ax.spines[spine].set_linewidth(0.9)
        ax.spines[spine].set_color('#2f2f2f')

    ax.set_axisbelow(True)
    ax.set_yscale('log')
    ax.scatter(
        n_vals,
        S_global_plot,
        s=70,
        color='C0',
        marker='o',
        edgecolors='white',
        linewidths=0.75,
        alpha=0.95,
        label=r'$\sum_{k > N} \sigma_k^2$',
        zorder=3,
    )
    ax.scatter(
        n_vals,
        S_sup_plot,
        s=70,
        color='C1',
        marker='s',
        edgecolors='white',
        linewidths=0.75,
        alpha=0.95,
        label=r'$\sup_{(\mu, t) \in \Theta \times [0, T]} \sum_{k > N} \sigma(\mu, t)_k^2$',
        zorder=3,
    )

    ax.set_xlabel('$N$', labelpad=8)
    ax.set_xlim(0.5, n_max + 0.5)

    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1))
    ax.yaxis.set_minor_formatter(NullFormatter())

    ax.grid(which='major', color='#c9c9c9', linestyle='-', linewidth=0.6, alpha=0.9)
    ax.grid(which='minor', axis='y', color='#dddddd', linestyle=':', linewidth=0.5, alpha=0.95)
    ax.tick_params(axis='both', which='major', direction='out', length=3.5, width=0.8, color='#2f2f2f', pad=5)
    ax.tick_params(axis='both', which='minor', direction='out', length=2.0, width=0.6, color='#5a5a5a')

    legend = ax.legend(
        loc='upper right',
        bbox_to_anchor=(0.97, 0.80),
        frameon=True,
        handletextpad=0.6,
        labelspacing=0.6,
    )
    for text in legend.get_texts():
        text.set_color('#2f2f2f')

    plt.show()
