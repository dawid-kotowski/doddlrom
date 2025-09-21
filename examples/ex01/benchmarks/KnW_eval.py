import numpy as np
import matplotlib.pyplot as plt

path = 'examples/ex01/training_data/pod_singular_values_ex01.npz'
data = np.load(path)

sigma_global = data['sigma_global_NA'].astype(float)     # [N_A]
# Use precomputed sup if present; if not, sup over (mu,t) from sigma_mu_t
sigma_sup = (data['sigma_mu_t_sup'].astype(float)
             if 'sigma_mu_t_sup' in data
             else np.max(data['sigma_mu_t'].astype(float), axis=(0,1)))

sigma_global = np.maximum(sigma_global, 1e-16)
sigma_sup    = np.maximum(sigma_sup,    1e-16)

N_A = sigma_global.shape[0]
n_vals = np.arange(1, 10, dtype=int)

S_global = np.array([np.sum(sigma_global[n+1:]) for n in n_vals])
S_sup    = np.array([np.sum(sigma_sup[n+1:])    for n in n_vals])

# Fit window (skip a few points at both ends)
drop_head = max(2, int(0.02 * len(n_vals)))
drop_tail = max(2, int(0.02 * len(n_vals)))
fit_slice = slice(drop_head, len(n_vals)-drop_tail)

x_fit = n_vals[fit_slice].astype(float)
y_global_fit = np.maximum(S_global[fit_slice], 1e-16)
y_sup_fit    = np.maximum(S_sup[fit_slice],    1e-16)

# log–log linear regression: log S ≈ a + b log n, with alpha = -b
slope_g, intercept_g = np.polyfit(np.log(x_fit), np.log(y_global_fit), 1)
slope_s, intercept_s = np.polyfit(np.log(x_fit), np.log(y_sup_fit), 1)
alpha = -slope_g
beta  = -slope_s

print(f"Estimated alpha (global tail ~ n^(-alpha)): {alpha:.3f}")
print(f"Estimated beta  (sup tail ~ n^(-beta))   : {beta:.3f}")

# Fitted lines for plotting
Sg_fit_line = np.exp(intercept_g) * n_vals**slope_g
Ss_fit_line = np.exp(intercept_s) * n_vals**slope_s

# Plot 1: Global tail
plt.figure()
plt.loglog(n_vals, S_global, label='Global tail sum S(n)')
plt.loglog(n_vals, Sg_fit_line, linestyle='--', label=f'Fit: n^({slope_g:.2f}) → α≈{alpha:.2f}')
plt.xlabel('n'); plt.ylabel('Tail sum S(n)')
plt.title('Global singular-values tail (log-log)')
plt.legend(); plt.tight_layout(); plt.show()

# Plot 2: Supremum tail
plt.figure()
plt.loglog(n_vals, S_sup, label='Sup tail sum S_sup(n)')
plt.loglog(n_vals, Ss_fit_line, linestyle='--', label=f'Fit: n^({slope_s:.2f}) → β≈{beta:.2f}')
plt.xlabel('n'); plt.ylabel('Tail sum S_sup(n)')
plt.title('Sup over (μ,t) singular-values tail (log-log)')
plt.legend(); plt.tight_layout(); plt.show()
