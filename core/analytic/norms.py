import numpy as np

def g_norm(U, G):
    return np.sqrt(g_norm_2(U, G), dtype=np.float64)

def g_norm_2(U, G):
    if U.ndim == 1:
        val = float(U @ (G @ U))
        return max(val, 0.0)
    vals = np.einsum('ti,ij,tj->t', U, G, U, optimize=True)
    return np.maximum(vals, 0.0)
