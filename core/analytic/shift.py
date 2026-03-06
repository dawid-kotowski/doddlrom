from typing import Optional

import numpy as np
import torch
from utils.paths import training_data_path


def _load_shift(example_name: str):
    shift_path = training_data_path(example_name) / f"dirichlet_shift_{example_name}.npz"
    if not shift_path.exists():
        return None
    blob = np.load(shift_path)
    return blob["ut0"].astype(np.float32)


def _apply_shift(U: np.ndarray, ut0: Optional[np.ndarray]) -> np.ndarray:
    if ut0 is None:
        return U
    if ut0.ndim == 1:
        return U - ut0[None, :]
    if ut0.shape[0] == U.shape[0]:
        return U - ut0
    if ut0.shape[0] == 1:
        return U - ut0[0][None, :]
    # fallback: align on time dimension
    T = min(U.shape[0], ut0.shape[0])
    U_adj = U.copy()
    U_adj[:T] = U_adj[:T] - ut0[:T]
    return U_adj
