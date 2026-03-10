import numpy as np
import matplotlib.pyplot as plt
from utils.paths import training_data_path

example_name = 'ex01'

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


x_fit = n_vals.astype(float)
slope_g, intercept_g = fit_loglog(x_fit, S_global, fit_cutoff)
slope_s, intercept_s = fit_loglog(x_fit, S_sup, fit_cutoff)

alpha = -slope_g if np.isfinite(slope_g) else np.nan
beta = -slope_s if np.isfinite(slope_s) else np.nan

print(f"Estimated alpha (global tail ~ n^(-alpha)): {alpha:.3f}")
print(f"Estimated beta  (sup tail ~ n^(-beta))   : {beta:.3f}")

Sg_fit_line = np.exp(intercept_g) * x_fit ** slope_g if np.isfinite(slope_g) else None
Ss_fit_line = np.exp(intercept_s) * x_fit ** slope_s if np.isfinite(slope_s) else None

plt.figure()
plt.loglog(n_vals, np.maximum(S_global, eps), label='Global tail sum S(n)')
if Sg_fit_line is not None:
    plt.loglog(n_vals, np.maximum(Sg_fit_line, eps), linestyle='--', label=f'Fit: n^({slope_g:.2f}) -> alpha~{alpha:.2f}')
plt.xlabel('n')
plt.ylabel('Tail sum S(n)')
plt.title('Global singular-values tail (log-log)')
plt.xlim(1, n_plot_max)
plt.legend()
plt.tight_layout()
plt.show()

plt.figure()
plt.loglog(n_vals, np.maximum(S_sup, eps), label='Sup tail sum S_sup(n)')
if Ss_fit_line is not None:
    plt.loglog(n_vals, np.maximum(Ss_fit_line, eps), linestyle='--', label=f'Fit: n^({slope_s:.2f}) -> beta~{beta:.2f}')
plt.xlabel('n')
plt.ylabel('Tail sum S_sup(n)')
plt.title('Sup over (mu,t) singular-values tail (log-log)')
plt.xlim(1, n_plot_max)
plt.legend()
plt.tight_layout()
plt.show()
