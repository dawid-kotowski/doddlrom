"""
Parameters / hyperparameter registry for examples/.

This container is tailored to current constructors & trainers in
`reduced_order_models.py` (innerDOD, DFNN, HadamardNN, Encoder/Decoder,
statDOD, statHadamardNN, CoLoRA, and all *Trainer classes*).

Key goals
- Centralize all fixed problem sizes (N_h, N_A, N, N_prime, n, Nt, ...)
- Offer switchable presets that only change network layers (problem sizes unchanged)
- Provide *factory-style* kwargs that match constructors exactly

NEW (flexible AE config)
------------------------
For the autoencoders (POD and DOD), the fields
  - *_hidden_channels
  - *_kernel
  - *_stride
  - *_padding
may now be **either** a scalar (int) **or** a list[int] of length = num_layers.
This aligns with the flexible Encoder/Decoder you added: they accept ints or lists
and broadcast scalars internally.

Examples (you can put these inside any PRESETS entry):
    "pod_hidden_channels": [8,16,32],   # channels per conv block
    "pod_stride": [1,2,2],              # stride per block
    "pod_kernel": 5,                     # still fine as scalar
    "pod_padding": 2,                    # scalar or list
The kwargs emitted by make_*_Encoder/Decoder_kwargs are unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Union
import math

# ---------- Helper type alias for AE hyperparams (scalar or list) -----------
IntOrList = Union[int, List[int]]



# =============================================================================
# EXAMPLE 01
# =============================================================================

EX01_PRESETS: Dict[str, Dict[str, Any]] = {
    "baseline": {
        # DOD only
        "preprocess_dim": 2,
        "dod_structure": [64, 32],

        # DOD+DFNN
        "df_layers": [128, 64],

        # DOD-DL-ROM (AE on N' <-> n)
        "dod_dl_df_layers": [8, 4],
        "dod_in_channels": 1,
        "dod_hidden_channels": 8,
        "dod_kernel": 3,
        "dod_stride": 1,
        "dod_padding": 1,
        "dod_num_layers": 1,

        # POD-DL-ROM (AE on N <-> n)
        "pod_df_layers": [16, 8],
        "pod_in_channels": 1,
        "pod_hidden_channels": [8, 16],  
        "pod_kernel": [3, 3],
        "pod_stride": [1, 2], 
        "pod_padding": [1, 1],
        "pod_num_layers": 2,

        # Stationary + CoLoRA
        "L": 3,
        "stat_dod_structure": [128, 64],
        "stat_phi_n_structure": [16, 8, 4],
    },
    "test1": {
        "preprocess_dim": 2,
        "dod_structure": [32],
        "df_layers": [8],

        "dod_dl_df_layers": [8],
        "dod_in_channels": 1,
        "dod_hidden_channels": 2,
        "dod_kernel": 3, "dod_stride": 2, "dod_padding": 1,
        "dod_num_layers": 1,

        "pod_df_layers": [8],
        "pod_in_channels": 1,
        "pod_hidden_channels": 2,
        "pod_kernel": 3, "pod_stride": 2, "pod_padding": 1,
        "pod_num_layers": 1,

        "L": 2,
        "stat_dod_structure": [32],
        "stat_phi_n_structure": [8],
    },
    "test2": {
        "preprocess_dim": 2,
        "dod_structure": [64, 32],
        "df_layers": [16, 8],

        "dod_dl_df_layers": [16, 8],
        "dod_in_channels": 1,
        "dod_hidden_channels": 4,
        "dod_kernel": 3, "dod_stride": 2, "dod_padding": 1,
        "dod_num_layers": 2,

        "pod_df_layers": [16, 8],
        "pod_in_channels": 1,
        "pod_hidden_channels": 4,
        "pod_kernel": 3, "pod_stride": 2, "pod_padding": 1,
        "pod_num_layers": 2,

        "L": 3,
        "stat_dod_structure": [64, 32],
        "stat_phi_n_structure": [16, 8],
    },
    "test3": {
        "preprocess_dim": 2,
        "dod_structure": [128, 64],
        "df_layers": [32, 16],

        "dod_dl_df_layers": [32, 16],
        "dod_in_channels": 1,
        "dod_hidden_channels": 8,
        "dod_kernel": 5, "dod_stride": 2, "dod_padding": 2,
        "dod_num_layers": 2,

        "pod_df_layers": [32, 16],
        "pod_in_channels": 1,
        "pod_hidden_channels": 8,
        "pod_kernel": 5, "pod_stride": 2, "pod_padding": 2,
        "pod_num_layers": 3,

        "L": 3,
        "stat_dod_structure": [128, 64],
        "stat_phi_n_structure": [32, 16],
    },
    "test4": {
        "preprocess_dim": 2,
        "dod_structure": [128, 64, 64],
        "df_layers": [32, 32, 16],

        "dod_dl_df_layers": [32, 32, 16],
        "dod_in_channels": 1,
        "dod_hidden_channels": 16,
        "dod_kernel": 5, "dod_stride": 2, "dod_padding": 2,
        "dod_num_layers": 3,

        "pod_df_layers": [32, 32, 16],
        "pod_in_channels": 1,
        "pod_hidden_channels": 16,
        "pod_kernel": 5, "pod_stride": 2, "pod_padding": 2,
        "pod_num_layers": 3,

        "L": 4,
        "stat_dod_structure": [128, 64, 64],
        "stat_phi_n_structure": [32, 16, 16],
    },
    "test5": {
        "preprocess_dim": 3,
        "dod_structure": [256, 128, 64],
        "df_layers": [64, 32, 16],

        "dod_dl_df_layers": [64, 32, 16],
        "dod_in_channels": 1,
        "dod_hidden_channels": 24,
        "dod_kernel": 5, "dod_stride": 2, "dod_padding": 2,
        "dod_num_layers": 4,

        "pod_df_layers": [64, 32, 16],
        "pod_in_channels": 1,
        "pod_hidden_channels": 24,
        "pod_kernel": 5, "pod_stride": 2, "pod_padding": 2,
        "pod_num_layers": 4,

        "L": 4,
        "stat_dod_structure": [256, 128, 64],
        "stat_phi_n_structure": [64, 32, 16],
    },
    "test6": {
        "preprocess_dim": 2,
        "dod_structure": [160, 80, 64],
        "df_layers": [40, 24, 12],

        "dod_dl_df_layers": [40, 24, 12],
        "dod_in_channels": 1,
        "dod_hidden_channels": 18,
        "dod_kernel": [5, 5, 5], "dod_stride": [1, 2, 2], "dod_padding": [2, 2, 2],
        "dod_num_layers": 3,

        "pod_df_layers": [40, 24, 12],
        "pod_in_channels": 1,
        "pod_hidden_channels": 18,
        "pod_kernel": [5, 5, 5], "pod_stride": [1, 2, 2], "pod_padding": [2, 2, 2],
        "pod_num_layers": 3,

        "L": 4,
        "stat_dod_structure": [160, 80, 64],
        "stat_phi_n_structure": [40, 24, 12],
    },
    "test7": {
        "preprocess_dim": 2,
        "dod_structure": [192, 96, 64],
        "df_layers": [48, 28, 14],

        "dod_dl_df_layers": [48, 28, 14],
        "dod_in_channels": 1,
        "dod_hidden_channels": 20,
        "dod_kernel": [5, 5, 5], "dod_stride": [1, 2, 2], "dod_padding": [2, 2, 2],
        "dod_num_layers": 3,

        "pod_df_layers": [48, 28, 14],
        "pod_in_channels": 1,
        "pod_hidden_channels": 20,
        "pod_kernel": [5, 5, 5], "pod_stride": [1, 2, 2], "pod_padding": [2, 2, 2],
        "pod_num_layers": 3,

        "L": 4,
        "stat_dod_structure": [192, 96, 64],
        "stat_phi_n_structure": [48, 28, 14],
    },
    "test8": {
        "preprocess_dim": 3,
        "dod_structure": [224, 112, 64],
        "df_layers": [56, 32, 16],

        "dod_dl_df_layers": [56, 32, 16],
        "dod_in_channels": 1,
        "dod_hidden_channels": 20,
        "dod_kernel": [5, 5, 5], "dod_stride": [1, 2, 2], "dod_padding": [2, 2, 2],
        "dod_num_layers": 3,

        "pod_df_layers": [56, 32, 16],
        "pod_in_channels": 1,
        "pod_hidden_channels": 20,
        "pod_kernel": [5, 5, 5], "pod_stride": [1, 2, 2], "pod_padding": [2, 2, 2],
        "pod_num_layers": 3,

        "L": 4,
        "stat_dod_structure": [224, 112, 64],
        "stat_phi_n_structure": [56, 32, 16],
    },
    "test9": {
        "preprocess_dim": 3,
        "dod_structure": [240, 120, 64],
        "df_layers": [60, 32, 16],

        "dod_dl_df_layers": [60, 32, 16],
        "dod_in_channels": 1,
        "dod_hidden_channels": 22,
        "dod_kernel": [5, 5, 5], "dod_stride": [1, 2, 2], "dod_padding": [2, 2, 2],
        "dod_num_layers": 3,

        "pod_df_layers": [60, 32, 16],
        "pod_in_channels": 1,
        "pod_hidden_channels": 22,
        "pod_kernel": [5, 5, 5], "pod_stride": [1, 2, 2], "pod_padding": [2, 2, 2],
        "pod_num_layers": 3,

        "L": 4,
        "stat_dod_structure": [240, 120, 64],
        "stat_phi_n_structure": [60, 32, 16],
    },
    "test10": {
        "preprocess_dim": 3,
        "dod_structure": [256, 128, 64],
        "df_layers": [64, 32, 16],

        "dod_dl_df_layers": [64, 32, 16],
        "dod_in_channels": 1,
        "dod_hidden_channels": 22,
        "dod_kernel": [5, 5, 5, 5], "dod_stride": [1, 1, 2, 2], "dod_padding": [2, 2, 2, 2],
        "dod_num_layers": 4,

        "pod_df_layers": [64, 32, 16],
        "pod_in_channels": 1,
        "pod_hidden_channels": 22,
        "pod_kernel": [5, 5, 5, 5], "pod_stride": [1, 1, 2, 2], "pod_padding": [2, 2, 2, 2],
        "pod_num_layers": 4,

        "L": 4,
        "stat_dod_structure": [256, 128, 64],
        "stat_phi_n_structure": [64, 32, 16],
    },
    "test11": {
        "preprocess_dim": 3,
        "dod_structure": [256, 128, 64],
        "df_layers": [64, 36, 18],

        "dod_dl_df_layers": [64, 36, 18],
        "dod_in_channels": 1,
        "dod_hidden_channels": 24,
        "dod_kernel": [5, 5, 5, 5], "dod_stride": [1, 2, 1, 2], "dod_padding": [2, 2, 2, 2],
        "dod_num_layers": 4,

        "pod_df_layers": [64, 36, 18],
        "pod_in_channels": 1,
        "pod_hidden_channels": 24,
        "pod_kernel": [5, 5, 5, 5], "pod_stride": [1, 2, 1, 2], "pod_padding": [2, 2, 2, 2],
        "pod_num_layers": 4,

        "L": 4,
        "stat_dod_structure": [256, 128, 64],
        "stat_phi_n_structure": [64, 36, 18],
    },
    "test12": {
        "preprocess_dim": 3,
        "dod_structure": [256, 128, 64],
        "df_layers": [64, 38, 18],

        "dod_dl_df_layers": [64, 38, 18],
        "dod_in_channels": 1,
        "dod_hidden_channels": 24,
        "dod_kernel": [5, 5, 5, 5], "dod_stride": [1, 1, 2, 2], "dod_padding": [2, 2, 2, 2],
        "dod_num_layers": 4,

        "pod_df_layers": [64, 38, 18],
        "pod_in_channels": 1,
        "pod_hidden_channels": 24,
        "pod_kernel": [5, 5, 5, 5], "pod_stride": [1, 1, 2, 2], "pod_padding": [2, 2, 2, 2],
        "pod_num_layers": 4,

        "L": 4,
        "stat_dod_structure": [256, 128, 64],
        "stat_phi_n_structure": [64, 38, 18],
    },
    "test13": {
        "preprocess_dim": 3,
        "dod_structure": [256, 128, 64],
        "df_layers": [64, 38, 20],

        "dod_dl_df_layers": [64, 38, 20],
        "dod_in_channels": 1,
        "dod_hidden_channels": 24,
        "dod_kernel": [5, 5, 5, 5], "dod_stride": [1, 2, 2, 1], "dod_padding": [2, 2, 2, 2],
        "dod_num_layers": 4,

        "pod_df_layers": [64, 38, 20],
        "pod_in_channels": 1,
        "pod_hidden_channels": 24,
        "pod_kernel": [5, 5, 5, 5], "pod_stride": [1, 2, 2, 1], "pod_padding": [2, 2, 2, 2],
        "pod_num_layers": 4,

        "L": 4,
        "stat_dod_structure": [256, 128, 64],
        "stat_phi_n_structure": [64, 38, 20],
    },
    "test14": {
        "preprocess_dim": 3,
        "dod_structure": [256, 128, 64],
        "df_layers": [64, 40, 20],

        "dod_dl_df_layers": [64, 40, 20],
        "dod_in_channels": 1,
        "dod_hidden_channels": 24,
        "dod_kernel": [5, 5, 5, 5], "dod_stride": [1, 1, 2, 2], "dod_padding": [2, 2, 2, 2],
        "dod_num_layers": 4,

        "pod_df_layers": [64, 40, 20],
        "pod_in_channels": 1,
        "pod_hidden_channels": 24,
        "pod_kernel": [5, 5, 5, 5], "pod_stride": [1, 1, 2, 2], "pod_padding": [2, 2, 2, 2],
        "pod_num_layers": 4,

        "L": 4,
        "stat_dod_structure": [256, 128, 64],
        "stat_phi_n_structure": [64, 40, 20],
    },
    "test15": {
        "preprocess_dim": 3,
        "dod_structure": [256, 128, 64],
        "df_layers": [64, 40, 20],

        "dod_dl_df_layers": [64, 40, 20],
        "dod_in_channels": 1,
        "dod_hidden_channels": 24,
        "dod_kernel": [5, 5, 5, 5], "dod_stride": [1, 2, 1, 2], "dod_padding": [2, 2, 2, 2],
        "dod_num_layers": 4,

        "pod_df_layers": [64, 40, 20],
        "pod_in_channels": 1,
        "pod_hidden_channels": 24,
        "pod_kernel": [5, 5, 5, 5], "pod_stride": [1, 2, 1, 2], "pod_padding": [2, 2, 2, 2],
        "pod_num_layers": 4,

        "L": 4,
        "stat_dod_structure": [256, 128, 64],
        "stat_phi_n_structure": [64, 40, 20],
    },
}


@dataclass
class Ex01Parameters:
    # -------- Fixed example sizes (problem-level) ----------------------------
    N_A: int = 64
    N: int = 32
    N_prime: int = 8
    n: int = 2
    Nt: int = 30
    Ns: int = 100
    T: float = 0.009
    diameter: float = 0.11
    parameter_mu_dim: int = 1
    parameter_nu_dim: int = 1

    # -------- innerDOD ------------------------------------------------------
    preprocess_dim: int = 2
    dod_structure: List[int] = field(default_factory=lambda: [32, 16])

    # -------- DOD+DFNN ------------------------------------------------------
    df_layers: List[int] = field(default_factory=lambda: [16, 8])

    # -------- DOD-DL-ROM (AE on N' <-> n) -----------------------------------
    dod_dl_df_layers: List[int] = field(default_factory=lambda: [32, 16, 8])
    dod_in_channels: int = 1
    dod_hidden_channels: IntOrList = 1        
    dod_kernel: IntOrList = 3                 
    dod_stride: IntOrList = 2                 
    dod_padding: IntOrList = 1                
    dod_num_layers: int = 1

    # -------- POD-DL-ROM (AE on N_A <-> n) ----------------------------------
    pod_df_layers: List[int] = field(default_factory=lambda: [32, 16, 8])
    pod_in_channels: int = 1
    pod_hidden_channels: IntOrList = 1        
    pod_kernel: IntOrList = 3                 
    pod_stride: IntOrList = 2                 
    pod_padding: IntOrList = 1                
    pod_num_layers: int = 3

    # -------- Stationary + CoLoRA ------------------------------------------
    L: int = 3
    stat_m: int = 4
    stat_dod_structure: List[int] = field(default_factory=lambda: [128, 64])
    stat_phi_n_structure: List[int] = field(default_factory=lambda: [16, 8])

    # -------- Training defaults ---------------------------------------------
    generalepochs: int = 300
    generalrestarts: int = 3
    generalpatience: int = 40

    dod_epochs: int = 1000
    dod_restarts: int = 2
    dod_patience: int = 40

    # -------- Meta -----------------------------------------------------------
    profile: str = "baseline"

    def __post_init__(self) -> None:
        preset = EX01_PRESETS.get(self.profile, {})
        for k, v in preset.items():
            if hasattr(self, k):
                setattr(self, k, v)

    # ----------------- Sanity checks ----------------------------------------
    def assert_consistent(self) -> None:
        if not (self.N_A >= self.N >= self.n > 0):
            raise ValueError(f"Require N_A >= N >= n > 0, got N_A={self.N_A}, N={self.N}, n={self.n}")
        if self.N_prime <= 0 or self.Nt <= 0 or self.diameter <= 0:
            raise ValueError("N_prime, Nt, diameter must be positive")
        for lst_name in [
            "dod_structure", "df_layers", "dod_dl_df_layers",
            "pod_df_layers", "stat_dod_structure", "stat_phi_n_structure"
        ]:
            lst = getattr(self, lst_name)
            if not all(isinstance(x, int) and x > 0 for x in lst):
                raise ValueError(f"All entries in {lst_name} must be positive integers, got {lst}")

    # ----------------- Factory kwargs --------------------------------------
    def make_innerDOD_kwargs(self) -> Dict[str, Any]:
        return dict(
            seed_dim=self.preprocess_dim,
            geometric_dim=self.parameter_mu_dim,
            root_layer_sizes=list(self.dod_structure),
            N_prime=self.N_prime,
            N_A=self.N_A,
        )

    def make_dod_dfnn_DFNN_kwargs(self) -> Dict[str, Any]:
        return dict(
            geometric_dim=self.parameter_mu_dim,
            physical_dim=self.parameter_nu_dim,
            n=self.N_prime,
            layer_sizes=list(self.df_layers),
        )

    def make_dod_dfnn_Hadamard_kwargs(self) -> Dict[str, Any]:
        return dict(
            geometric_dim=self.parameter_mu_dim,
            physical_dim=self.parameter_nu_dim,
            m_0=self.preprocess_dim,
            n_0=self.N_prime,
            layer_sizes=list(self.dod_structure),
        )

    def make_dod_dl_DFNN_kwargs(self) -> Dict[str, Any]:
        return dict(
            geometric_dim=self.parameter_mu_dim,
            physical_dim=self.parameter_nu_dim,
            n=self.n,
            layer_sizes=list(self.dod_dl_df_layers),
        )

    def make_dod_dl_Encoder_kwargs(self) -> Dict[str, Any]:
        return dict(
            input_dim=self.N_prime,
            in_channels=self.dod_in_channels,
            hidden_channels=self.dod_hidden_channels,  # int or list
            latent_dim=self.n,
            num_layers=self.dod_num_layers,
            kernel=self.dod_kernel,                    # int or list
            stride=self.dod_stride,                    # int or list
            padding=self.dod_padding,                  # int or list
        )

    def make_dod_dl_Decoder_kwargs(self) -> Dict[str, Any]:
        return dict(
            output_dim=self.N_prime,
            out_channels=self.dod_in_channels,
            hidden_channels=self.dod_hidden_channels,  # int or list
            latent_dim=self.n,
            num_layers=self.dod_num_layers,
            kernel=self.dod_kernel,                    # int or list
            stride=self.dod_stride,                    # int or list
            padding=self.dod_padding,                  # int or list
        )

    def make_pod_DFNN_kwargs(self) -> Dict[str, Any]:
        return dict(
            geometric_dim=self.parameter_mu_dim,
            physical_dim=self.parameter_nu_dim,
            n=self.n,
            layer_sizes=list(self.pod_df_layers),
        )

    def make_pod_Encoder_kwargs(self) -> Dict[str, Any]:
        return dict(
            input_dim=self.N,
            in_channels=self.pod_in_channels,
            hidden_channels=self.pod_hidden_channels,  # int or list
            latent_dim=self.n,
            num_layers=self.pod_num_layers,
            kernel=self.pod_kernel,                    # int or list
            stride=self.pod_stride,                    # int or list
            padding=self.pod_padding,                  # int or list
        )

    def make_pod_Decoder_kwargs(self) -> Dict[str, Any]:
        return dict(
            output_dim=self.N,
            out_channels=self.pod_in_channels,
            hidden_channels=self.pod_hidden_channels,  # int or list
            latent_dim=self.n,
            num_layers=self.pod_num_layers,
            kernel=self.pod_kernel,                    # int or list
            stride=self.pod_stride,                    # int or list
            padding=self.pod_padding,                  # int or list
        )

    def make_statDOD_kwargs(self) -> Dict[str, Any]:
        return dict(
            geometric_dim=self.parameter_mu_dim,
            seed_dim=self.preprocess_dim,
            num_roots=self.N_prime,
            root_output_dim=self.N,
            root_layer_sizes=list(self.stat_dod_structure),
        )

    def make_statHadamard_kwargs(self) -> Dict[str, Any]:
        return dict(
            param_mu_space_dim=self.parameter_mu_dim,
            param_nu_space_dim=self.parameter_nu_dim,
            m_0=self.stat_m,
            n_0=self.N_prime,
            layer_sizes=list(self.stat_phi_n_structure),
        )

    def make_CoLoRA_kwargs(self) -> Dict[str, Any]:
        return dict(
            out_dim=self.N,
            L=self.L,
            dyn_dim=self.N_prime,
            physical_dim=self.parameter_nu_dim,
            with_bias=True,
        )

    def trainer_defaults(self) -> Dict[str, Any]:
        return dict(
            epochs=self.generalepochs,
            restarts=self.generalrestarts,
            patience=self.generalpatience,
        )
    
    def dod_trainer_defaults(self) -> Dict[str, Any]:
        return dict(
            epochs=self.dod_epochs,
            restarts=self.dod_restarts,
            patience=self.dod_patience
        )



# =============================================================================
# EXAMPLE 02
# =============================================================================

EX02_PRESETS: Dict[str, Dict[str, Any]] = {
    "baseline": {
        "preprocess_dim": 2,
        "dod_structure": [400, 200, 100],
        "df_layers": [67, 33, 17],

        # DOD-DL-ROM (AE on N'<->n)
        "dod_dl_df_layers": [34, 17, 9],
        "dod_in_channels": 1,
        "dod_hidden_channels": 33,
        "dod_kernel": 3, "dod_stride": 1, "dod_padding": 1,
        "dod_num_layers": 1,  

        # POD-DL-ROM
        "pod_df_layers": [100, 50, 25],
        "pod_in_channels": 1,
        "pod_hidden_channels": [40, 80],
        "pod_kernel": [3, 3], "pod_stride": [1, 2], "pod_padding": [1, 1],
        "pod_num_layers": 2,

        "L": 3,
        "stat_dod_structure": [256, 128],
        "stat_phi_n_structure": [32, 16, 8],
    },
    "test1": {
        "preprocess_dim": 2,
        "dod_structure": [24, 12],
        "df_layers": [8, 4],
        "dod_dl_df_layers": [4, 4],
        "dod_in_channels": 1,
        "dod_hidden_channels": 4,
        "dod_kernel": 3, "dod_stride": 1, "dod_padding": 1,
        "dod_num_layers": 1,
        "pod_df_layers": [12, 6],
        "pod_in_channels": 1,
        "pod_hidden_channels": [4, 8],
        "pod_kernel": [3, 3], "pod_stride": [1, 2], "pod_padding": [1, 1],
        "pod_num_layers": 2,
        "L": 3,
        "stat_dod_structure": [128, 64],
        "stat_phi_n_structure": [16, 8, 8],
    },
    "test2": {
        "preprocess_dim": 2,
        "dod_structure": [28, 14],
        "df_layers": [9, 5],
        "dod_dl_df_layers": [5, 4],
        "dod_in_channels": 1,
        "dod_hidden_channels": 5,
        "dod_kernel": 3, "dod_stride": 1, "dod_padding": 1,
        "dod_num_layers": 1,
        "pod_df_layers": [14, 7],
        "pod_in_channels": 1,
        "pod_hidden_channels": [5, 9],
        "pod_kernel": [3, 3], "pod_stride": [1, 2], "pod_padding": [1, 1],
        "pod_num_layers": 2,
        "L": 3,
        "stat_dod_structure": [148, 74],
        "stat_phi_n_structure": [18, 9, 5],
    },
    "test3": {
        "preprocess_dim": 2,
        "dod_structure": [32, 16],
        "df_layers": [11, 5],
        "dod_dl_df_layers": [6, 4],
        "dod_in_channels": 1,
        "dod_hidden_channels": 5,
        "dod_kernel": 3, "dod_stride": 1, "dod_padding": 1,
        "dod_num_layers": 1,
        "pod_df_layers": [16, 8],
        "pod_in_channels": 1,
        "pod_hidden_channels": [5, 11],
        "pod_kernel": [3, 3], "pod_stride": [1, 2], "pod_padding": [1, 1],
        "pod_num_layers": 2,
        "L": 3,
        "stat_dod_structure": [170, 85],
        "stat_phi_n_structure": [22, 11, 5],
    },
    "test4": {
        "preprocess_dim": 2,
        "dod_structure": [37, 19],
        "df_layers": [12, 6],
        "dod_dl_df_layers": [6, 4],
        "dod_in_channels": 1,
        "dod_hidden_channels": 6,
        "dod_kernel": 3, "dod_stride": 1, "dod_padding": 1,
        "dod_num_layers": 1,
        "pod_df_layers": [19, 10],
        "pod_in_channels": 1,
        "pod_hidden_channels": [6, 12],
        "pod_kernel": [3, 3], "pod_stride": [1, 2], "pod_padding": [1, 1],
        "pod_num_layers": 2,
        "L": 3,
        "stat_dod_structure": [196, 101],
        "stat_phi_n_structure": [24, 12, 6],
    },
    "test5": {
        "preprocess_dim": 2,
        "dod_structure": [42, 21],
        "df_layers": [14, 7],
        "dod_dl_df_layers": [7, 4],
        "dod_in_channels": 1,
        "dod_hidden_channels": 7,
        "dod_kernel": 3, "dod_stride": 1, "dod_padding": 1,
        "dod_num_layers": 1,
        "pod_df_layers": [21, 11],
        "pod_in_channels": 1,
        "pod_hidden_channels": [7, 14],
        "pod_kernel": [3, 3], "pod_stride": [1, 2], "pod_padding": [1, 1],
        "pod_num_layers": 2,
        "L": 3,
        "stat_dod_structure": [223, 111],
        "stat_phi_n_structure": [28, 14, 7],
    },
    "test6": {
        "preprocess_dim": 2,
        "dod_structure": [48, 24],
        "df_layers": [16, 8],
        "dod_dl_df_layers": [8, 4],
        "dod_in_channels": 1,
        "dod_hidden_channels": 8,
        "dod_kernel": 3, "dod_stride": 1, "dod_padding": 1,
        "dod_num_layers": 1,
        "pod_df_layers": [24, 12],
        "pod_in_channels": 1,
        "pod_hidden_channels": [8, 16],
        "pod_kernel": [3, 3], "pod_stride": [1, 2], "pod_padding": [1, 1],
        "pod_num_layers": 2,
        "L": 3,
        "stat_dod_structure": [254, 127],
        "stat_phi_n_structure": [32, 16, 8],
    },
    "test7": {
        "preprocess_dim": 2,
        "dod_structure": [54, 27],
        "df_layers": [18, 9],
        "dod_dl_df_layers": [9, 5],
        "dod_in_channels": 1,
        "dod_hidden_channels": 9,
        "dod_kernel": 3, "dod_stride": 1, "dod_padding": 1,
        "dod_num_layers": 1,
        "pod_df_layers": [27, 14],
        "pod_in_channels": 1,
        "pod_hidden_channels": [9, 18],
        "pod_kernel": [3, 3], "pod_stride": [1, 2], "pod_padding": [1, 1],
        "pod_num_layers": 2,
        "L": 3,
        "stat_dod_structure": [286, 143],
        "stat_phi_n_structure": [36, 18, 9],
    },
    "test8": {
        "preprocess_dim": 2,
        "dod_structure": [61, 31],
        "df_layers": [20, 10],
        "dod_dl_df_layers": [10, 5],
        "dod_in_channels": 1,
        "dod_hidden_channels": 10,
        "dod_kernel": 3, "dod_stride": 1, "dod_padding": 1,
        "dod_num_layers": 1,
        "pod_df_layers": [31, 16],
        "pod_in_channels": 1,
        "pod_hidden_channels": [10, 20],
        "pod_kernel": [3, 3], "pod_stride": [1, 2], "pod_padding": [1, 1],
        "pod_num_layers": 2,
        "L": 3,
        "stat_dod_structure": [323, 164],
        "stat_phi_n_structure": [40, 20, 10],
    },
    "test9": {
        "preprocess_dim": 2,
        "dod_structure": [68, 34],
        "df_layers": [23, 11],
        "dod_dl_df_layers": [12, 6],
        "dod_in_channels": 1,
        "dod_hidden_channels": 11,
        "dod_kernel": 3, "dod_stride": 1, "dod_padding": 1,
        "dod_num_layers": 1,
        "pod_df_layers": [34, 17],
        "pod_in_channels": 1,
        "pod_hidden_channels": [11, 23],
        "pod_kernel": [3, 3], "pod_stride": [1, 2], "pod_padding": [1, 1],
        "pod_num_layers": 2,
        "L": 3,
        "stat_dod_structure": [360, 180],
        "stat_phi_n_structure": [46, 23, 11],
    },
    "test10": {
        "preprocess_dim": 2,
        "dod_structure": [76, 38],
        "df_layers": [25, 13],
        "dod_dl_df_layers": [13, 7],
        "dod_in_channels": 1,
        "dod_hidden_channels": 13,
        "dod_kernel": 3, "dod_stride": 1, "dod_padding": 1,
        "dod_num_layers": 1,
        "pod_df_layers": [38, 19],
        "pod_in_channels": 1,
        "pod_hidden_channels": [13, 25],
        "pod_kernel": [3, 3], "pod_stride": [1, 2], "pod_padding": [1, 1],
        "pod_num_layers": 2,
        "L": 3,
        "stat_dod_structure": [403, 201],
        "stat_phi_n_structure": [50, 25, 13],
    },
    "test11": {
        "preprocess_dim": 2,
        "dod_structure": [84, 42],
        "df_layers": [28, 14],
        "dod_dl_df_layers": [14, 7],
        "dod_in_channels": 1,
        "dod_hidden_channels": 14,
        "dod_kernel": 3, "dod_stride": 1, "dod_padding": 1,
        "dod_num_layers": 1,
        "pod_df_layers": [42, 21],
        "pod_in_channels": 1,
        "pod_hidden_channels": [14, 28],
        "pod_kernel": [3, 3], "pod_stride": [1, 2], "pod_padding": [1, 1],
        "pod_num_layers": 2,
        "L": 3,
        "stat_dod_structure": [445, 223],
        "stat_phi_n_structure": [56, 28, 14],
    },
    "test12": {
        "preprocess_dim": 2,
        "dod_structure": [93, 47],
        "df_layers": [31, 16],
        "dod_dl_df_layers": [16, 8],
        "dod_in_channels": 1,
        "dod_hidden_channels": 16,
        "dod_kernel": 3, "dod_stride": 1, "dod_padding": 1,
        "dod_num_layers": 1,
        "pod_df_layers": [47, 24],
        "pod_in_channels": 1,
        "pod_hidden_channels": [16, 31],
        "pod_kernel": [3, 3], "pod_stride": [1, 2], "pod_padding": [1, 1],
        "pod_num_layers": 2,
        "L": 3,
        "stat_dod_structure": [493, 249],
        "stat_phi_n_structure": [62, 31, 16],
    },
    "test13": {
        "preprocess_dim": 2,
        "dod_structure": [103, 52],
        "df_layers": [34, 17],
        "dod_dl_df_layers": [17, 9],
        "dod_in_channels": 1,
        "dod_hidden_channels": 17,
        "dod_kernel": 3, "dod_stride": 1, "dod_padding": 1,
        "dod_num_layers": 1,
        "pod_df_layers": [52, 26],
        "pod_in_channels": 1,
        "pod_hidden_channels": [17, 34],
        "pod_kernel": [3, 3], "pod_stride": [1, 2], "pod_padding": [1, 1],
        "pod_num_layers": 2,
        "L": 3,
        "stat_dod_structure": [546, 276],
        "stat_phi_n_structure": [68, 34, 17],
    },
    "test14": {
        "preprocess_dim": 2,
        "dod_structure": [114, 57],
        "df_layers": [38, 19],
        "dod_dl_df_layers": [19, 10],
        "dod_in_channels": 1,
        "dod_hidden_channels": 19,
        "dod_kernel": 3, "dod_stride": 1, "dod_padding": 1,
        "dod_num_layers": 1,
        "pod_df_layers": [57, 29],
        "pod_in_channels": 1,
        "pod_hidden_channels": [19, 38],
        "pod_kernel": [3, 3], "pod_stride": [1, 2], "pod_padding": [1, 1],
        "pod_num_layers": 2,
        "L": 3,
        "stat_dod_structure": [604, 302],
        "stat_phi_n_structure": [76, 38, 19],
    },
    "test15": {
        "preprocess_dim": 2,
        "dod_structure": [126, 63],
        "df_layers": [42, 21],
        "dod_dl_df_layers": [21, 11],
        "dod_in_channels": 1,
        "dod_hidden_channels": 21,
        "dod_kernel": 3, "dod_stride": 1, "dod_padding": 1,
        "dod_num_layers": 1,
        "pod_df_layers": [63, 32],
        "pod_in_channels": 1,
        "pod_hidden_channels": [21, 42],
        "pod_kernel": [3, 3], "pod_stride": [1, 2], "pod_padding": [1, 1],
        "pod_num_layers": 2,
        "L": 3,
        "stat_dod_structure": [668, 334],
        "stat_phi_n_structure": [84, 42, 21],
    },
    "test16": {
        "preprocess_dim": 2,
        "dod_structure": [139, 70, 35],
        "df_layers": [46, 23, 12],
        "dod_dl_df_layers": [23, 12, 6],
        "dod_in_channels": 1,
        "dod_hidden_channels": 23,
        "dod_kernel": 3, "dod_stride": 1, "dod_padding": 1,
        "dod_num_layers": 1,
        "pod_df_layers": [70, 35, 18],
        "pod_in_channels": 1,
        "pod_hidden_channels": [23, 46],
        "pod_kernel": [3, 3], "pod_stride": [1, 2], "pod_padding": [1, 1],
        "pod_num_layers": 2,
        "L": 3,
        "stat_dod_structure": [737, 371],
        "stat_phi_n_structure": [92, 46, 23],
    },
    "test17": {
        "preprocess_dim": 2,
        "dod_structure": [153, 77, 39],
        "df_layers": [51, 26, 13],
        "dod_dl_df_layers": [26, 13, 7],
        "dod_in_channels": 1,
        "dod_hidden_channels": 26,
        "dod_kernel": 3, "dod_stride": 1, "dod_padding": 1,
        "dod_num_layers": 1,
        "pod_df_layers": [77, 39, 20],
        "pod_in_channels": 1,
        "pod_hidden_channels": [26, 51],
        "pod_kernel": [3, 3], "pod_stride": [1, 2], "pod_padding": [1, 1],
        "pod_num_layers": 2,
        "L": 3,
        "stat_dod_structure": [811, 408],
        "stat_phi_n_structure": [102, 51, 26],
    },
    "test18": {
        "preprocess_dim": 2,
        "dod_structure": [168, 84, 42],
        "df_layers": [56, 28, 14],
        "dod_dl_df_layers": [28, 14, 7],
        "dod_in_channels": 1,
        "dod_hidden_channels": 28,
        "dod_kernel": 3, "dod_stride": 1, "dod_padding": 1,
        "dod_num_layers": 1,
        "pod_df_layers": [84, 42, 21],
        "pod_in_channels": 1,
        "pod_hidden_channels": [28, 56],
        "pod_kernel": [3, 3], "pod_stride": [1, 2], "pod_padding": [1, 1],
        "pod_num_layers": 2,
        "L": 3,
        "stat_dod_structure": [890, 445],
        "stat_phi_n_structure": [112, 56, 28],
    },
    "test19": {
        "preprocess_dim": 2,
        "dod_structure": [184, 92, 46],
        "df_layers": [61, 31, 15],
        "dod_dl_df_layers": [31, 16, 8],
        "dod_in_channels": 1,
        "dod_hidden_channels": 31,
        "dod_kernel": 3, "dod_stride": 1, "dod_padding": 1,
        "dod_num_layers": 1,
        "pod_df_layers": [92, 46, 23],
        "pod_in_channels": 1,
        "pod_hidden_channels": [31, 61],
        "pod_kernel": [3, 3], "pod_stride": [1, 2], "pod_padding": [1, 1],
        "pod_num_layers": 2,
        "L": 3,
        "stat_dod_structure": [975, 488],
        "stat_phi_n_structure": [122, 61, 31],
    },
    "test20": {
        "preprocess_dim": 2,
        "dod_structure": [200, 100, 50],
        "df_layers": [67, 33, 17],
        "dod_dl_df_layers": [34, 17, 9],
        "dod_in_channels": 1,
        "dod_hidden_channels": 33,
        "dod_kernel": 3, "dod_stride": 1, "dod_padding": 1,
        "dod_num_layers": 1,
        "pod_df_layers": [100, 50, 25],
        "pod_in_channels": 1,
        "pod_hidden_channels": [33, 67],
        "pod_kernel": [3, 3], "pod_stride": [1, 2], "pod_padding": [1, 1],
        "pod_num_layers": 2,
        "L": 3,
        "stat_dod_structure": [1060, 530],
        "stat_phi_n_structure": [134, 67, 33],
    },
}


@dataclass
class Ex02Parameters:
    # -------- Fixed example sizes (problem-level) ----------------------------
    N_A: int = 32
    N: int = 16
    N_prime: int = 4
    n: int = 2
    Nt: int = 20
    Ns: int = 4**4
    T: int = 1.
    diameter: float = 0.03
    parameter_mu_dim: int = 3
    parameter_nu_dim: int = 1

    # -------- innerDOD ------------------------------------------------------
    preprocess_dim: int = 4
    dod_structure: List[int] = field(default_factory=lambda: [32, 16])

    # -------- DOD+DFNN ------------------------------------------------------
    df_layers: List[int] = field(default_factory=lambda: [16, 8])

    # -------- DOD-DL-ROM (AE on N' <-> n + DFNN on n <-> p+q+1) -------------
    dod_dl_df_layers: List[int] = field(default_factory=lambda: [32, 16, 8])
    dod_in_channels: int = 1
    dod_hidden_channels: IntOrList = 1
    dod_lin_dim_ae: int = 0
    dod_kernel: IntOrList = 3
    dod_stride: IntOrList = 2
    dod_padding: IntOrList = 1
    dod_num_layers: int = 1

    # -------- POD-DL-ROM (AE on N <-> n + DFNN on n <-> p+q+1) -------------
    pod_df_layers: List[int] = field(default_factory=lambda: [32, 16, 8])
    pod_in_channels: int = 1
    pod_hidden_channels: IntOrList = 1
    pod_lin_dim_ae: int = 0
    pod_kernel: IntOrList = 3
    pod_stride: IntOrList = 2
    pod_padding: IntOrList = 1
    pod_num_layers: int = 3

    # -------- Meta -----------------------------------------------------------
    profile: str = "baseline"

    def __post_init__(self) -> None:
        preset = EX03_PRESETS.get(self.profile, {})
        for k, v in preset.items():
            if hasattr(self, k):
                setattr(self, k, v)

    # -------- Stationary + CoLoRA ------------------------------------------
    L: int = 3
    stat_m: int = 8
    stat_dod_structure: List[int] = field(default_factory=lambda: [128, 64])
    stat_phi_n_structure: List[int] = field(default_factory=lambda: [16, 8])

    # -------- Training defaults ---------------------------------------------
    generalepochs: int = 30
    generalrestarts: int = 3
    generalpatience: int = 10

    dod_epochs: int = 90
    dod_restarts: int = 3
    dod_patience: int = 10

    # -------- Meta -----------------------------------------------------------
    profile: str = "baseline"

    def __post_init__(self) -> None:
        preset = EX02_PRESETS.get(self.profile, {})
        for k, v in preset.items():
            if hasattr(self, k):
                setattr(self, k, v)

    # ----------------- Sanity checks ----------------------------------------
    def assert_consistent(self) -> None:
        if not (self.N_A >= self.N >= self.n > 0):
            raise ValueError(f"Require N_A >= N >= n > 0, got N_A={self.N_A}, N={self.N}, n={self.n}")
        if self.N_prime <= 0 or self.Nt <= 0 or self.diameter <= 0:
            raise ValueError("N_prime, Nt, diameter must be positive")
        for lst_name in [
            "dod_structure", "df_layers", "dod_dl_df_layers",
            "pod_df_layers", "stat_dod_structure", "stat_phi_n_structure"
        ]:
            lst = getattr(self, lst_name)
            if not all(isinstance(x, int) and x > 0 for x in lst):
                raise ValueError(f"All entries in {lst_name} must be positive integers, got {lst}")

    # ----------------- Factory kwargs --------------------------------------
    def make_innerDOD_kwargs(self) -> Dict[str, Any]:
        return dict(
            seed_dim=self.preprocess_dim,
            geometric_dim=self.parameter_mu_dim,
            root_layer_sizes=list(self.dod_structure),
            N_prime=self.N_prime,
            N_A=self.N_A,
        )

    def make_dod_dfnn_DFNN_kwargs(self) -> Dict[str, Any]:
        return dict(
            geometric_dim=self.parameter_mu_dim,
            physical_dim=self.parameter_nu_dim,
            n=self.N_prime,
            layer_sizes=list(self.df_layers),
        )

    def make_dod_dfnn_Hadamard_kwargs(self) -> Dict[str, Any]:
        return dict(
            geometric_dim=self.parameter_mu_dim,
            physical_dim=self.parameter_nu_dim,
            m_0=self.preprocess_dim,
            n_0=self.N_prime,
            layer_sizes=list(self.dod_structure),
        )

    def make_dod_dl_DFNN_kwargs(self) -> Dict[str, Any]:
        return dict(
            geometric_dim=self.parameter_mu_dim,
            physical_dim=self.parameter_nu_dim,
            n=self.n,
            layer_sizes=list(self.dod_dl_df_layers),
        )

    def make_dod_dl_Encoder_kwargs(self) -> Dict[str, Any]:
        return dict(
            input_dim=self.N_prime,
            in_channels=self.dod_in_channels,
            hidden_channels=self.dod_hidden_channels,
            latent_dim=self.n,
            num_layers=self.dod_num_layers,
            kernel=self.dod_kernel,
            stride=self.dod_stride,
            padding=self.dod_padding,
        )

    def make_dod_dl_Decoder_kwargs(self) -> Dict[str, Any]:
        return dict(
            output_dim=self.N_prime,
            out_channels=self.dod_in_channels,
            hidden_channels=self.dod_hidden_channels,
            latent_dim=self.n,
            num_layers=self.dod_num_layers,
            kernel=self.dod_kernel,
            stride=self.dod_stride,
            padding=self.dod_padding,
        )

    def make_pod_DFNN_kwargs(self) -> Dict[str, Any]:
        return dict(
            geometric_dim=self.parameter_mu_dim,
            physical_dim=self.parameter_nu_dim,
            n=self.n,
            layer_sizes=list(self.pod_df_layers),
        )

    def make_pod_Encoder_kwargs(self) -> Dict[str, Any]:
        return dict(
            input_dim=self.N,
            in_channels=self.pod_in_channels,
            hidden_channels=self.pod_hidden_channels,
            latent_dim=self.n,
            num_layers=self.pod_num_layers,
            kernel=self.pod_kernel,
            stride=self.pod_stride,
            padding=self.pod_padding,
        )

    def make_pod_Decoder_kwargs(self) -> Dict[str, Any]:
        return dict(
            output_dim=self.N,
            out_channels=self.pod_in_channels,
            hidden_channels=self.pod_hidden_channels,
            latent_dim=self.n,
            num_layers=self.pod_num_layers,
            kernel=self.pod_kernel,
            stride=self.pod_stride,
            padding=self.pod_padding,
        )

    def make_statDOD_kwargs(self) -> Dict[str, Any]:
        return dict(
            geometric_dim=self.parameter_mu_dim,
            seed_dim=self.preprocess_dim,
            num_roots=self.N_prime,
            root_output_dim=self.N,
            root_layer_sizes=list(self.stat_dod_structure),
        )

    def make_statHadamard_kwargs(self) -> Dict[str, Any]:
        return dict(
            param_mu_space_dim=self.parameter_mu_dim,
            param_nu_space_dim=self.parameter_nu_dim,
            m_0=self.stat_m,
            n_0=self.N_prime,
            layer_sizes=list(self.stat_phi_n_structure),
        )

    def make_CoLoRA_kwargs(self) -> Dict[str, Any]:
        return dict(
            out_dim=self.N,
            L=self.L,
            dyn_dim=self.N_prime,
            physical_dim=self.parameter_nu_dim,
            with_bias=True,
        )

    def trainer_defaults(self) -> Dict[str, Any]:
        return dict(
            epochs=self.generalepochs,
            restarts=self.generalrestarts,
            patience=self.generalpatience,
        )
    
    def dod_trainer_defaults(self) -> Dict[str, Any]:
        return dict(
            epochs=self.dod_epochs,
            restarts=self.dod_restarts,
            patience=self.dod_patience
        )



# =============================================================================
# EXAMPLE 03
# =============================================================================

EX03_PRESETS: Dict[str, Dict[str, Any]] = {
    "baseline": {
        "preprocess_dim": 2,
        "dod_structure": [32, 16],
        "df_layers": [32, 16],

        "dod_dl_df_layers": [24, 12],
        "dod_in_channels": 1,
        "dod_hidden_channels": 4,
        "dod_kernel": 3, "dod_stride": 1, "dod_padding": 1,
        "dod_num_layers": 1,

        "pod_df_layers": [96, 48],
        "pod_in_channels": 1,
        "pod_hidden_channels": 4,
        "pod_kernel": 3, "pod_stride": 1, "pod_padding": 1,
        "pod_num_layers": 1,

        "L": 3,
        "stat_dod_structure": [64, 32],
        "stat_phi_n_structure": [16, 8],
    },
    "test1": {
        "preprocess_dim": 1,
        "dod_structure": [16],
        "df_layers": [8],
        "dod_dl_df_layers": [8],
        "dod_in_channels": 1,
        "dod_hidden_channels": 2,
        "dod_kernel": 3, "dod_stride": 2, "dod_padding": 1,
        "dod_num_layers": 1,
        "pod_df_layers": [8],
        "pod_in_channels": 1,
        "pod_hidden_channels": 2,
        "pod_kernel": 3, "pod_stride": 2, "pod_padding": 1,
        "pod_num_layers": 2,
        "L": 2,
        "stat_dod_structure": [16],
        "stat_phi_n_structure": [8],
    },
    "test2": {
        "preprocess_dim": 2,
        "dod_structure": [32, 16],
        "df_layers": [12, 8],
        "dod_dl_df_layers": [12, 8],
        "dod_in_channels": 1,
        "dod_hidden_channels": 4,
        "dod_kernel": 3, "dod_stride": 2, "dod_padding": 1,
        "dod_num_layers": 2,
        "pod_df_layers": [12, 8],
        "pod_in_channels": 1,
        "pod_hidden_channels": 4,
        "pod_kernel": 3, "pod_stride": 2, "pod_padding": 1,
        "pod_num_layers": 3,
        "L": 3,
        "stat_dod_structure": [32, 16],
        "stat_phi_n_structure": [12, 8],
    },
    "test3": {
        "preprocess_dim": 2,
        "dod_structure": [64, 32],
        "df_layers": [16, 12],
        "dod_dl_df_layers": [16, 12],
        "dod_in_channels": 1,
        "dod_hidden_channels": 8,
        "dod_kernel": 5, "dod_stride": 2, "dod_padding": 2,
        "dod_num_layers": 2,   
        "pod_df_layers": [16, 12],
        "pod_in_channels": 1,
        "pod_hidden_channels": 8,
        "pod_kernel": 5, "pod_stride": 2, "pod_padding": 2,
        "pod_num_layers": 3,  
        "L": 3,
        "stat_dod_structure": [64, 32],
        "stat_phi_n_structure": [16, 12],
    },
    "test4": {
        "preprocess_dim": 2,
        "dod_structure": [96, 64, 32],
        "df_layers": [24, 16, 8],
        "dod_dl_df_layers": [24, 16, 8],
        "dod_in_channels": 1,
        "dod_hidden_channels": 12,
        "dod_kernel": 5, "dod_stride": 2, "dod_padding": 2,
        "dod_num_layers": 3,
        "pod_df_layers": [24, 16, 8],
        "pod_in_channels": 1,
        "pod_hidden_channels": 12,
        "pod_kernel": 5, "pod_stride": 2, "pod_padding": 2,
        "pod_num_layers": 4,
        "L": 4,
        "stat_dod_structure": [96, 64, 32],
        "stat_phi_n_structure": [24, 16, 8],
    },
    "test5": {
        "preprocess_dim": 3,
        "dod_structure": [128, 96, 48],
        "df_layers": [32, 24, 12],
        "dod_dl_df_layers": [32, 24, 12],
        "dod_in_channels": 1,
        "dod_hidden_channels": 16,
        "dod_kernel": 5, "dod_stride": 2, "dod_padding": 2,
        "dod_num_layers": 3,
        "pod_dl_layers": [32, 24, 12],
        "pod_in_channels": 1,
        "pod_hidden_channels": 16,
        "pod_kernel": 5, "pod_stride": 2, "pod_padding": 2,
        "pod_num_layers": 4,
        "L": 4,
        "stat_dod_structure": [128, 96, 48],
        "stat_phi_n_structure": [32, 24, 12],
    },
    "test6": {
        "preprocess_dim": 2,
        "dod_structure": [112, 72, 40],
        "df_layers": [28, 20, 10],
        "dod_dl_df_layers": [28, 20, 10],
        "dod_in_channels": 1,
        "dod_hidden_channels": 14,
        "dod_kernel": [5, 5, 5], "dod_stride": [1, 2, 2], "dod_padding": [2, 2, 2],
        "dod_num_layers": 3,
        "pod_df_layers": [28, 20, 10],
        "pod_in_channels": 1,
        "pod_hidden_channels": 14,
        "pod_kernel": [5, 5, 5, 5], "pod_stride": [1, 1, 2, 2], "pod_padding": [2, 2, 2, 2],
        "pod_num_layers": 4,
        "L": 4,
        "stat_dod_structure": [112, 72, 40],
        "stat_phi_n_structure": [28, 20, 10],
    },
    "test7": {
        "preprocess_dim": 2,
        "dod_structure": [120, 80, 44],
        "df_layers": [30, 20, 10],
        "dod_dl_df_layers": [30, 20, 10],
        "dod_in_channels": 1,
        "dod_hidden_channels": 14,
        "dod_kernel": [5, 5, 5], "dod_stride": [1, 2, 2], "dod_padding": [2, 2, 2],
        "dod_num_layers": 3,
        "pod_df_layers": [30, 20, 10],
        "pod_in_channels": 1,
        "pod_hidden_channels": 14,
        "pod_kernel": [5, 5, 5, 5], "pod_stride": [1, 1, 2, 2], "pod_padding": [2, 2, 2, 2],
        "pod_num_layers": 4,
        "L": 4,
        "stat_dod_structure": [120, 80, 44],
        "stat_phi_n_structure": [30, 20, 10],
    },
    "test8": {
        "preprocess_dim": 3,
        "dod_structure": [128, 84, 44],
        "df_layers": [32, 22, 11],
        "dod_dl_df_layers": [32, 22, 11],
        "dod_in_channels": 1,
        "dod_hidden_channels": 15,
        "dod_kernel": [5, 5, 5], "dod_stride": [1, 2, 2], "dod_padding": [2, 2, 2],
        "dod_num_layers": 3,
        "pod_df_layers": [32, 22, 11],
        "pod_in_channels": 1,
        "pod_hidden_channels": 15,
        "pod_kernel": [5, 5, 5, 5], "pod_stride": [1, 1, 2, 2], "pod_padding": [2, 2, 2, 2],
        "pod_num_layers": 4,
        "L": 4,
        "stat_dod_structure": [128, 84, 44],
        "stat_phi_n_structure": [32, 22, 11],
    },
    "test9": {
        "preprocess_dim": 3,
        "dod_structure": [128, 88, 46],
        "df_layers": [32, 22, 12],
        "dod_dl_df_layers": [32, 22, 12],
        "dod_in_channels": 1,
        "dod_hidden_channels": 15,
        "dod_kernel": [5, 5, 5], "dod_stride": [1, 2, 2], "dod_padding": [2, 2, 2],
        "dod_num_layers": 3,
        "pod_df_layers": [32, 22, 12],
        "pod_in_channels": 1,
        "pod_hidden_channels": 15,
        "pod_kernel": [5, 5, 5, 5], "pod_stride": [1, 1, 2, 2], "pod_padding": [2, 2, 2, 2],
        "pod_num_layers": 4,
        "L": 4,
        "stat_dod_structure": [128, 88, 46],
        "stat_phi_n_structure": [32, 22, 12],
    },
    "test10": {
        "preprocess_dim": 3,
        "dod_structure": [128, 92, 46],
        "df_layers": [32, 24, 12],
        "dod_dl_df_layers": [32, 24, 12],
        "dod_in_channels": 1,
        "dod_hidden_channels": 15,
        "dod_kernel": [5, 5, 5], "dod_stride": [1, 2, 2], "dod_padding": [2, 2, 2],
        "dod_num_layers": 3,
        "pod_df_layers": [32, 24, 12],
        "pod_in_channels": 1,
        "pod_hidden_channels": 15,
        "pod_kernel": [5, 5, 5, 5], "pod_stride": [1, 1, 2, 2], "pod_padding": [2, 2, 2, 2],
        "pod_num_layers": 4,
        "L": 4,
        "stat_dod_structure": [128, 92, 46],
        "stat_phi_n_structure": [32, 24, 12],
    },
    "test11": {
        "preprocess_dim": 3,
        "dod_structure": [128, 96, 48],
        "df_layers": [32, 24, 12],
        "dod_dl_df_layers": [32, 24, 12],
        "dod_in_channels": 1,
        "dod_hidden_channels": 16,
        "dod_kernel": [5, 5, 5], "dod_stride": [1, 2, 2], "dod_padding": [2, 2, 2],
        "dod_num_layers": 3,
        "pod_df_layers": [32, 24, 12],
        "pod_in_channels": 1,
        "pod_hidden_channels": 16,
        "pod_kernel": [5, 5, 5, 5], "pod_stride": [1, 1, 2, 2], "pod_padding": [2, 2, 2, 2],
        "pod_num_layers": 4,
        "L": 4,
        "stat_dod_structure": [128, 96, 48],
        "stat_phi_n_structure": [32, 24, 12],
    },
    "test12": {
        "preprocess_dim": 3,
        "dod_structure": [128, 96, 48],
        "df_layers": [34, 24, 12],
        "dod_dl_df_layers": [34, 24, 12],
        "dod_in_channels": 1,
        "dod_hidden_channels": 16,
        "dod_kernel": [5, 5, 5], "dod_stride": [1, 2, 2], "dod_padding": [2, 2, 2],
        "dod_num_layers": 3,
        "pod_df_layers": [34, 24, 12],
        "pod_in_channels": 1,
        "pod_hidden_channels": 16,
        "pod_kernel": [5, 5, 5, 5], "pod_stride": [1, 1, 2, 2], "pod_padding": [2, 2, 2, 2],
        "pod_num_layers": 4,
        "L": 4,
        "stat_dod_structure": [128, 96, 48],
        "stat_phi_n_structure": [34, 24, 12],
    },
    "test13": {
        "preprocess_dim": 3,
        "dod_structure": [128, 96, 48],
        "df_layers": [34, 26, 13],
        "dod_dl_df_layers": [34, 26, 13],
        "dod_in_channels": 1,
        "dod_hidden_channels": 16,
        "dod_kernel": [5, 5, 5], "dod_stride": [1, 2, 2], "dod_padding": [2, 2, 2],
        "dod_num_layers": 3,
        "pod_df_layers": [34, 26, 13],
        "pod_in_channels": 1,
        "pod_hidden_channels": 16,
        "pod_kernel": [5, 5, 5, 5], "pod_stride": [1, 1, 2, 2], "pod_padding": [2, 2, 2, 2],
        "pod_num_layers": 4,
        "L": 4,
        "stat_dod_structure": [128, 96, 48],
        "stat_phi_n_structure": [34, 26, 13],
    },
    "test14": {
        "preprocess_dim": 3,
        "dod_structure": [128, 96, 48],
        "df_layers": [36, 26, 13],
        "dod_dl_df_layers": [36, 26, 13],
        "dod_in_channels": 1,
        "dod_hidden_channels": 16,
        "dod_kernel": [5, 5, 5], "dod_stride": [1, 2, 2], "dod_padding": [2, 2, 2],
        "dod_num_layers": 3,
        "pod_df_layers": [36, 26, 13],
        "pod_in_channels": 1,
        "pod_hidden_channels": 16,
        "pod_kernel": [5, 5, 5, 5], "pod_stride": [1, 1, 2, 2], "pod_padding": [2, 2, 2, 2],
        "pod_num_layers": 4,
        "L": 4,
        "stat_dod_structure": [128, 96, 48],
        "stat_phi_n_structure": [36, 26, 13],
    },
    "test15": {
        "preprocess_dim": 3,
        "dod_structure": [128, 96, 48],
        "df_layers": [36, 28, 14],
        "dod_dl_df_layers": [36, 28, 14],
        "dod_in_channels": 1,
        "dod_hidden_channels": 16,
        "dod_kernel": [5, 5, 5], "dod_stride": [1, 2, 2], "dod_padding": [2, 2, 2],
        "dod_num_layers": 3,
        "pod_df_layers": [36, 28, 14],
        "pod_in_channels": 1,
        "pod_hidden_channels": 16,
        "pod_kernel": [5, 5, 5, 5], "pod_stride": [1, 1, 2, 2], "pod_padding": [2, 2, 2, 2],
        "pod_num_layers": 4,
        "L": 4,
        "stat_dod_structure": [128, 96, 48],
        "stat_phi_n_structure": [36, 28, 14],
    },
}


@dataclass
class Ex03Parameters:
    # -------- Fixed example sizes (problem-level) ----------------------------
    N_A: int = 101
    N: int = 4        # already did 32, 16, 8
    N_prime: int = 2  # already did 16, 8, 4
    n: int = 4
    Nt: int = 30
    Ns: int = 900
    T: float = 3.0
    diameter: float = 0.01
    parameter_mu_dim: int = 1
    parameter_nu_dim: int = 1

    # -------- innerDOD ------------------------------------------------------
    preprocess_dim: int = 2
    dod_structure: List[int] = field(default_factory=lambda: [32, 16])

    # -------- DOD+DFNN ------------------------------------------------------
    df_layers: List[int] = field(default_factory=lambda: [16, 8])

    # -------- DOD-DL-ROM (AE on N' <-> n + DFNN on n <-> p+q+1) -------------
    dod_dl_df_layers: List[int] = field(default_factory=lambda: [32, 16, 8])
    dod_in_channels: int = 1
    dod_hidden_channels: IntOrList = 1
    dod_lin_dim_ae: int = 0
    dod_kernel: IntOrList = 3
    dod_stride: IntOrList = 2
    dod_padding: IntOrList = 1
    dod_num_layers: int = 1

    # -------- POD-DL-ROM (AE on N <-> n + DFNN on n <-> p+q+1) -------------
    pod_df_layers: List[int] = field(default_factory=lambda: [32, 16, 8])
    pod_in_channels: int = 1
    pod_hidden_channels: IntOrList = 1
    pod_lin_dim_ae: int = 0
    pod_kernel: IntOrList = 3
    pod_stride: IntOrList = 2
    pod_padding: IntOrList = 1
    pod_num_layers: int = 3

    # -------- Stationary + CoLoRA ------------------------------------------
    L: int = 3
    stat_m: int = 8
    stat_dod_structure: List[int] = field(default_factory=lambda: [128, 64])
    stat_phi_n_structure: List[int] = field(default_factory=lambda: [16, 8])

    # -------- Training defaults ---------------------------------------------
    generalepochs: int = 200
    generalrestarts: int = 1
    generalpatience: int = 40

    dod_epochs: int = 300
    dod_restarts: int = 1
    dod_patience: int = 40

    # -------- Meta -----------------------------------------------------------
    profile: str = "baseline"

    def __post_init__(self) -> None:
        preset = EX03_PRESETS.get(self.profile, {})
        for k, v in preset.items():
            if hasattr(self, k):
                setattr(self, k, v)

    # ----------------- Sanity checks ----------------------------------------
    def assert_consistent(self) -> None:
        if not (self.N_A >= self.N >= self.n > 0):
            raise ValueError(f"Require N_A >= N >= n > 0, got N_A={self.N_A}, N={self.N}, n={self.n}")
        if self.N_prime <= 0 or self.Nt <= 0 or self.diameter <= 0:
            raise ValueError("N_prime, Nt, diameter must be positive")
        for lst_name in [
            "dod_structure", "df_layers", "dod_dl_df_layers",
            "pod_df_layers", "stat_dod_structure", "stat_phi_n_structure"
        ]:
            lst = getattr(self, lst_name)
            if not all(isinstance(x, int) and x > 0 for x in lst):
                raise ValueError(f"All entries in {lst_name} must be positive integers, got {lst}")

    # ----------------- Factory kwargs --------------------------------------
    def make_innerDOD_kwargs(self) -> Dict[str, Any]:
        return dict(
            seed_dim=self.preprocess_dim,
            geometric_dim=self.parameter_mu_dim,
            root_layer_sizes=list(self.dod_structure),
            N_prime=self.N_prime,
            N_A=self.N_A,
        )

    def make_dod_dfnn_DFNN_kwargs(self) -> Dict[str, Any]:
        return dict(
            geometric_dim=self.parameter_mu_dim,
            physical_dim=self.parameter_nu_dim,
            n=self.N_prime,
            layer_sizes=list(self.df_layers),
        )

    def make_dod_dfnn_Hadamard_kwargs(self) -> Dict[str, Any]:
        return dict(
            geometric_dim=self.parameter_mu_dim,
            physical_dim=self.parameter_nu_dim,
            m_0=self.preprocess_dim,
            n_0=self.N_prime,
            layer_sizes=list(self.dod_structure),
        )

    def make_dod_dl_DFNN_kwargs(self) -> Dict[str, Any]:
        return dict(
            geometric_dim=self.parameter_mu_dim,
            physical_dim=self.parameter_nu_dim,
            n=self.n,
            layer_sizes=list(self.dod_dl_df_layers),
        )

    def make_dod_dl_Encoder_kwargs(self) -> Dict[str, Any]:
        return dict(
            input_dim=self.N_prime,
            in_channels=self.dod_in_channels,
            hidden_channels=self.dod_hidden_channels,
            latent_dim=self.n,
            num_layers=self.dod_num_layers,
            kernel=self.dod_kernel,
            stride=self.dod_stride,
            padding=self.dod_padding,
        )

    def make_dod_dl_Decoder_kwargs(self) -> Dict[str, Any]:
        return dict(
            output_dim=self.N_prime,
            out_channels=self.dod_in_channels,
            hidden_channels=self.dod_hidden_channels,
            latent_dim=self.n,
            num_layers=self.dod_num_layers,
            kernel=self.dod_kernel,
            stride=self.dod_stride,
            padding=self.dod_padding,
        )

    def make_pod_DFNN_kwargs(self) -> Dict[str, Any]:
        return dict(
            geometric_dim=self.parameter_mu_dim,
            physical_dim=self.parameter_nu_dim,
            n=self.n,
            layer_sizes=list(self.pod_df_layers),
        )

    def make_pod_Encoder_kwargs(self) -> Dict[str, Any]:
        return dict(
            input_dim=self.N,
            in_channels=self.pod_in_channels,
            hidden_channels=self.pod_hidden_channels,
            latent_dim=self.n,
            num_layers=self.pod_num_layers,
            kernel=self.pod_kernel,
            stride=self.pod_stride,
            padding=self.pod_padding,
        )

    def make_pod_Decoder_kwargs(self) -> Dict[str, Any]:
        return dict(
            output_dim=self.N,
            out_channels=self.pod_in_channels,
            hidden_channels=self.pod_hidden_channels,
            latent_dim=self.n,
            num_layers=self.pod_num_layers,
            kernel=self.pod_kernel,
            stride=self.pod_stride,
            padding=self.pod_padding,
        )

    def make_statDOD_kwargs(self) -> Dict[str, Any]:
        return dict(
            geometric_dim=self.parameter_mu_dim,
            seed_dim=self.preprocess_dim,
            num_roots=self.N_prime,
            root_output_dim=self.N,
            root_layer_sizes=list(self.stat_dod_structure),
        )

    def make_statHadamard_kwargs(self) -> Dict[str, Any]:
        return dict(
            param_mu_space_dim=self.parameter_mu_dim,
            param_nu_space_dim=self.parameter_nu_dim,
            m_0=self.stat_m,
            n_0=self.N_prime,
            layer_sizes=list(self.stat_phi_n_structure),
        )

    def make_CoLoRA_kwargs(self) -> Dict[str, Any]:
        return dict(
            out_dim=self.N,
            L=self.L,
            dyn_dim=self.N_prime,
            physical_dim=self.parameter_nu_dim,
            with_bias=True,
        )

    def trainer_defaults(self) -> Dict[str, Any]:
        return dict(
            epochs=self.generalepochs,
            restarts=self.generalrestarts,
            patience=self.generalpatience,
        )

    def dod_trainer_defaults(self) -> Dict[str, Any]:
        return dict(
            epochs=self.dod_epochs,
            restarts=self.dod_restarts,
            patience=self.dod_patience
        )



# =============================================================================
# EXAMPLE 04
# =============================================================================

EX04_PRESETS: Dict[str, Dict[str, Any]] = {
    "baseline": {
        "preprocess_dim": 2,
        "dod_structure": [200, 100],
        "df_layers": [16, 8],

        "dod_dl_df_layers": [16, 8],
        "dod_in_channels": 1,
        "dod_hidden_channels": 8,
        "dod_kernel": 3, "dod_stride": 2, "dod_padding": 1,
        "dod_num_layers": 2,

        "pod_df_layers": [16, 8],
        "pod_in_channels": 1,
        "pod_hidden_channels": 8,
        "pod_kernel": 3, "pod_stride": 2, "pod_padding": 1,
        "pod_num_layers": 3,

        "L": 3,
        "stat_dod_structure": [64, 32],
        "stat_phi_n_structure": [16, 8],
    },
    "test1": {
        "preprocess_dim": 1,
        "dod_structure": [16],
        "df_layers": [8],
        "dod_dl_df_layers": [8],
        "dod_in_channels": 1,
        "dod_hidden_channels": 2,
        "dod_kernel": 3, "dod_stride": 2, "dod_padding": 1,
        "dod_num_layers": 1,
        "pod_df_layers": [8],
        "pod_in_channels": 1,
        "pod_hidden_channels": 2,
        "pod_kernel": 3, "pod_stride": 2, "pod_padding": 1,
        "pod_num_layers": 2,
        "L": 2,
        "stat_dod_structure": [16],
        "stat_phi_n_structure": [8],
    },
    "test2": {
        "preprocess_dim": 2,
        "dod_structure": [32, 16],
        "df_layers": [12, 8],
        "dod_dl_df_layers": [12, 8],
        "dod_in_channels": 1,
        "dod_hidden_channels": 4,
        "dod_kernel": 3, "dod_stride": 2, "dod_padding": 1,
        "dod_num_layers": 2,
        "pod_df_layers": [12, 8],
        "pod_in_channels": 1,
        "pod_hidden_channels": 4,
        "pod_kernel": 3, "pod_stride": 2, "pod_padding": 1,
        "pod_num_layers": 3,
        "L": 3,
        "stat_dod_structure": [32, 16],
        "stat_phi_n_structure": [12, 8],
    },
    "test3": {
        "preprocess_dim": 2,
        "dod_structure": [64, 32],
        "df_layers": [16, 12],
        "dod_dl_df_layers": [16, 12],
        "dod_in_channels": 1,
        "dod_hidden_channels": 8,
        "dod_kernel": 5, "dod_stride": 2, "dod_padding": 2,
        "dod_num_layers": 2,   
        "pod_df_layers": [16, 12],
        "pod_in_channels": 1,
        "pod_hidden_channels": 8,
        "pod_kernel": 5, "pod_stride": 2, "pod_padding": 2,
        "pod_num_layers": 3,  
        "L": 3,
        "stat_dod_structure": [64, 32],
        "stat_phi_n_structure": [16, 12],
    },
    "test4": {
        "preprocess_dim": 2,
        "dod_structure": [96, 64, 32],
        "df_layers": [24, 16, 8],
        "dod_dl_df_layers": [24, 16, 8],
        "dod_in_channels": 1,
        "dod_hidden_channels": 12,
        "dod_kernel": 5, "dod_stride": 2, "dod_padding": 2,
        "dod_num_layers": 3,
        "pod_df_layers": [24, 16, 8],
        "pod_in_channels": 1,
        "pod_hidden_channels": 12,
        "pod_kernel": 5, "pod_stride": 2, "pod_padding": 2,
        "pod_num_layers": 4,
        "L": 4,
        "stat_dod_structure": [96, 64, 32],
        "stat_phi_n_structure": [24, 16, 8],
    },
    "test5": {
        "preprocess_dim": 3,
        "dod_structure": [128, 96, 48],
        "df_layers": [32, 24, 12],
        "dod_dl_df_layers": [32, 24, 12],
        "dod_in_channels": 1,
        "dod_hidden_channels": 16,
        "dod_kernel": 5, "dod_stride": 2, "dod_padding": 2,
        "dod_num_layers": 3,
        "pod_dl_layers": [32, 24, 12],
        "pod_in_channels": 1,
        "pod_hidden_channels": 16,
        "pod_kernel": 5, "pod_stride": 2, "pod_padding": 2,
        "pod_num_layers": 4,
        "L": 4,
        "stat_dod_structure": [128, 96, 48],
        "stat_phi_n_structure": [32, 24, 12],
    },
    "test6": {
        "preprocess_dim": 2,
        "dod_structure": [112, 72, 40],
        "df_layers": [28, 20, 10],
        "dod_dl_df_layers": [28, 20, 10],
        "dod_in_channels": 1,
        "dod_hidden_channels": 14,
        "dod_kernel": [5, 5, 5], "dod_stride": [1, 2, 2], "dod_padding": [2, 2, 2],
        "dod_num_layers": 3,
        "pod_df_layers": [28, 20, 10],
        "pod_in_channels": 1,
        "pod_hidden_channels": 14,
        "pod_kernel": [5, 5, 5, 5], "pod_stride": [1, 1, 2, 2], "pod_padding": [2, 2, 2, 2],
        "pod_num_layers": 4,
        "L": 4,
        "stat_dod_structure": [112, 72, 40],
        "stat_phi_n_structure": [28, 20, 10],
    },
    "test7": {
        "preprocess_dim": 2,
        "dod_structure": [120, 80, 44],
        "df_layers": [30, 20, 10],
        "dod_dl_df_layers": [30, 20, 10],
        "dod_in_channels": 1,
        "dod_hidden_channels": 14,
        "dod_kernel": [5, 5, 5], "dod_stride": [1, 2, 2], "dod_padding": [2, 2, 2],
        "dod_num_layers": 3,
        "pod_df_layers": [30, 20, 10],
        "pod_in_channels": 1,
        "pod_hidden_channels": 14,
        "pod_kernel": [5, 5, 5, 5], "pod_stride": [1, 1, 2, 2], "pod_padding": [2, 2, 2, 2],
        "pod_num_layers": 4,
        "L": 4,
        "stat_dod_structure": [120, 80, 44],
        "stat_phi_n_structure": [30, 20, 10],
    },
    "test8": {
        "preprocess_dim": 3,
        "dod_structure": [128, 84, 44],
        "df_layers": [32, 22, 11],
        "dod_dl_df_layers": [32, 22, 11],
        "dod_in_channels": 1,
        "dod_hidden_channels": 15,
        "dod_kernel": [5, 5, 5], "dod_stride": [1, 2, 2], "dod_padding": [2, 2, 2],
        "dod_num_layers": 3,
        "pod_df_layers": [32, 22, 11],
        "pod_in_channels": 1,
        "pod_hidden_channels": 15,
        "pod_kernel": [5, 5, 5, 5], "pod_stride": [1, 1, 2, 2], "pod_padding": [2, 2, 2, 2],
        "pod_num_layers": 4,
        "L": 4,
        "stat_dod_structure": [128, 84, 44],
        "stat_phi_n_structure": [32, 22, 11],
    },
    "test9": {
        "preprocess_dim": 3,
        "dod_structure": [128, 88, 46],
        "df_layers": [32, 22, 12],
        "dod_dl_df_layers": [32, 22, 12],
        "dod_in_channels": 1,
        "dod_hidden_channels": 15,
        "dod_kernel": [5, 5, 5], "dod_stride": [1, 2, 2], "dod_padding": [2, 2, 2],
        "dod_num_layers": 3,
        "pod_df_layers": [32, 22, 12],
        "pod_in_channels": 1,
        "pod_hidden_channels": 15,
        "pod_kernel": [5, 5, 5, 5], "pod_stride": [1, 1, 2, 2], "pod_padding": [2, 2, 2, 2],
        "pod_num_layers": 4,
        "L": 4,
        "stat_dod_structure": [128, 88, 46],
        "stat_phi_n_structure": [32, 22, 12],
    },
    "test10": {
        "preprocess_dim": 3,
        "dod_structure": [128, 92, 46],
        "df_layers": [32, 24, 12],
        "dod_dl_df_layers": [32, 24, 12],
        "dod_in_channels": 1,
        "dod_hidden_channels": 15,
        "dod_kernel": [5, 5, 5], "dod_stride": [1, 2, 2], "dod_padding": [2, 2, 2],
        "dod_num_layers": 3,
        "pod_df_layers": [32, 24, 12],
        "pod_in_channels": 1,
        "pod_hidden_channels": 15,
        "pod_kernel": [5, 5, 5, 5], "pod_stride": [1, 1, 2, 2], "pod_padding": [2, 2, 2, 2],
        "pod_num_layers": 4,
        "L": 4,
        "stat_dod_structure": [128, 92, 46],
        "stat_phi_n_structure": [32, 24, 12],
    },
    "test11": {
        "preprocess_dim": 3,
        "dod_structure": [128, 96, 48],
        "df_layers": [32, 24, 12],
        "dod_dl_df_layers": [32, 24, 12],
        "dod_in_channels": 1,
        "dod_hidden_channels": 16,
        "dod_kernel": [5, 5, 5], "dod_stride": [1, 2, 2], "dod_padding": [2, 2, 2],
        "dod_num_layers": 3,
        "pod_df_layers": [32, 24, 12],
        "pod_in_channels": 1,
        "pod_hidden_channels": 16,
        "pod_kernel": [5, 5, 5, 5], "pod_stride": [1, 1, 2, 2], "pod_padding": [2, 2, 2, 2],
        "pod_num_layers": 4,
        "L": 4,
        "stat_dod_structure": [128, 96, 48],
        "stat_phi_n_structure": [32, 24, 12],
    },
    "test12": {
        "preprocess_dim": 3,
        "dod_structure": [128, 96, 48],
        "df_layers": [34, 24, 12],
        "dod_dl_df_layers": [34, 24, 12],
        "dod_in_channels": 1,
        "dod_hidden_channels": 16,
        "dod_kernel": [5, 5, 5], "dod_stride": [1, 2, 2], "dod_padding": [2, 2, 2],
        "dod_num_layers": 3,
        "pod_df_layers": [34, 24, 12],
        "pod_in_channels": 1,
        "pod_hidden_channels": 16,
        "pod_kernel": [5, 5, 5, 5], "pod_stride": [1, 1, 2, 2], "pod_padding": [2, 2, 2, 2],
        "pod_num_layers": 4,
        "L": 4,
        "stat_dod_structure": [128, 96, 48],
        "stat_phi_n_structure": [34, 24, 12],
    },
    "test13": {
        "preprocess_dim": 3,
        "dod_structure": [128, 96, 48],
        "df_layers": [34, 26, 13],
        "dod_dl_df_layers": [34, 26, 13],
        "dod_in_channels": 1,
        "dod_hidden_channels": 16,
        "dod_kernel": [5, 5, 5], "dod_stride": [1, 2, 2], "dod_padding": [2, 2, 2],
        "dod_num_layers": 3,
        "pod_df_layers": [34, 26, 13],
        "pod_in_channels": 1,
        "pod_hidden_channels": 16,
        "pod_kernel": [5, 5, 5, 5], "pod_stride": [1, 1, 2, 2], "pod_padding": [2, 2, 2, 2],
        "pod_num_layers": 4,
        "L": 4,
        "stat_dod_structure": [128, 96, 48],
        "stat_phi_n_structure": [34, 26, 13],
    },
    "test14": {
        "preprocess_dim": 3,
        "dod_structure": [128, 96, 48],
        "df_layers": [36, 26, 13],
        "dod_dl_df_layers": [36, 26, 13],
        "dod_in_channels": 1,
        "dod_hidden_channels": 16,
        "dod_kernel": [5, 5, 5], "dod_stride": [1, 2, 2], "dod_padding": [2, 2, 2],
        "dod_num_layers": 3,
        "pod_df_layers": [36, 26, 13],
        "pod_in_channels": 1,
        "pod_hidden_channels": 16,
        "pod_kernel": [5, 5, 5, 5], "pod_stride": [1, 1, 2, 2], "pod_padding": [2, 2, 2, 2],
        "pod_num_layers": 4,
        "L": 4,
        "stat_dod_structure": [128, 96, 48],
        "stat_phi_n_structure": [36, 26, 13],
    },
    "test15": {
        "preprocess_dim": 3,
        "dod_structure": [128, 96, 48],
        "df_layers": [36, 28, 14],
        "dod_dl_df_layers": [36, 28, 14],
        "dod_in_channels": 1,
        "dod_hidden_channels": 16,
        "dod_kernel": [5, 5, 5], "dod_stride": [1, 2, 2], "dod_padding": [2, 2, 2],
        "dod_num_layers": 3,
        "pod_df_layers": [36, 28, 14],
        "pod_in_channels": 1,
        "pod_hidden_channels": 16,
        "pod_kernel": [5, 5, 5, 5], "pod_stride": [1, 1, 2, 2], "pod_padding": [2, 2, 2, 2],
        "pod_num_layers": 4,
        "L": 4,
        "stat_dod_structure": [128, 96, 48],
        "stat_phi_n_structure": [36, 28, 14],
    },
}


@dataclass
class Ex04Parameters:
    # -------- Fixed example sizes (problem-level) ----------------------------
    N_A: int = 101
    N: int = 4        # already did 32, 16, 8
    N_prime: int = 2  # already did 16, 8, 4
    n: int = 4
    Nt: int = 30
    Ns: int = 900
    T: float = 3.0
    diameter: float = 0.01
    parameter_mu_dim: int = 1
    parameter_nu_dim: int = 1

    # -------- innerDOD ------------------------------------------------------
    preprocess_dim: int = 2
    dod_structure: List[int] = field(default_factory=lambda: [32, 16])

    # -------- DOD+DFNN ------------------------------------------------------
    df_layers: List[int] = field(default_factory=lambda: [16, 8])

    # -------- DOD-DL-ROM (AE on N' <-> n + DFNN on n <-> p+q+1) -------------
    dod_dl_df_layers: List[int] = field(default_factory=lambda: [32, 16, 8])
    dod_in_channels: int = 1
    dod_hidden_channels: IntOrList = 1
    dod_lin_dim_ae: int = 0
    dod_kernel: IntOrList = 3
    dod_stride: IntOrList = 2
    dod_padding: IntOrList = 1
    dod_num_layers: int = 1

    # -------- POD-DL-ROM (AE on N <-> n + DFNN on n <-> p+q+1) -------------
    pod_df_layers: List[int] = field(default_factory=lambda: [32, 16, 8])
    pod_in_channels: int = 1
    pod_hidden_channels: IntOrList = 1
    pod_lin_dim_ae: int = 0
    pod_kernel: IntOrList = 3
    pod_stride: IntOrList = 2
    pod_padding: IntOrList = 1
    pod_num_layers: int = 3

    # -------- Stationary + CoLoRA ------------------------------------------
    L: int = 3
    stat_m: int = 8
    stat_dod_structure: List[int] = field(default_factory=lambda: [128, 64])
    stat_phi_n_structure: List[int] = field(default_factory=lambda: [16, 8])

    # -------- Training defaults ---------------------------------------------
    generalepochs: int = 200
    generalrestarts: int = 1
    generalpatience: int = 40

    dod_epochs: int = 300
    dod_restarts: int = 1
    dod_patience: int = 40

    # -------- Meta -----------------------------------------------------------
    profile: str = "baseline"

    def __post_init__(self) -> None:
        preset = EX03_PRESETS.get(self.profile, {})
        for k, v in preset.items():
            if hasattr(self, k):
                setattr(self, k, v)

    # ----------------- Sanity checks ----------------------------------------
    def assert_consistent(self) -> None:
        if not (self.N_A >= self.N >= self.n > 0):
            raise ValueError(f"Require N_A >= N >= n > 0, got N_A={self.N_A}, N={self.N}, n={self.n}")
        if self.N_prime <= 0 or self.Nt <= 0 or self.diameter <= 0:
            raise ValueError("N_prime, Nt, diameter must be positive")
        for lst_name in [
            "dod_structure", "df_layers", "dod_dl_df_layers",
            "pod_df_layers", "stat_dod_structure", "stat_phi_n_structure"
        ]:
            lst = getattr(self, lst_name)
            if not all(isinstance(x, int) and x > 0 for x in lst):
                raise ValueError(f"All entries in {lst_name} must be positive integers, got {lst}")

    # ----------------- Factory kwargs --------------------------------------
    def make_innerDOD_kwargs(self) -> Dict[str, Any]:
        return dict(
            seed_dim=self.preprocess_dim,
            geometric_dim=self.parameter_mu_dim,
            root_layer_sizes=list(self.dod_structure),
            N_prime=self.N_prime,
            N_A=self.N_A,
        )

    def make_dod_dfnn_DFNN_kwargs(self) -> Dict[str, Any]:
        return dict(
            geometric_dim=self.parameter_mu_dim,
            physical_dim=self.parameter_nu_dim,
            n=self.N_prime,
            layer_sizes=list(self.df_layers),
        )

    def make_dod_dfnn_Hadamard_kwargs(self) -> Dict[str, Any]:
        return dict(
            geometric_dim=self.parameter_mu_dim,
            physical_dim=self.parameter_nu_dim,
            m_0=self.preprocess_dim,
            n_0=self.N_prime,
            layer_sizes=list(self.dod_structure),
        )

    def make_dod_dl_DFNN_kwargs(self) -> Dict[str, Any]:
        return dict(
            geometric_dim=self.parameter_mu_dim,
            physical_dim=self.parameter_nu_dim,
            n=self.n,
            layer_sizes=list(self.dod_dl_df_layers),
        )

    def make_dod_dl_Encoder_kwargs(self) -> Dict[str, Any]:
        return dict(
            input_dim=self.N_prime,
            in_channels=self.dod_in_channels,
            hidden_channels=self.dod_hidden_channels,
            latent_dim=self.n,
            num_layers=self.dod_num_layers,
            kernel=self.dod_kernel,
            stride=self.dod_stride,
            padding=self.dod_padding,
        )

    def make_dod_dl_Decoder_kwargs(self) -> Dict[str, Any]:
        return dict(
            output_dim=self.N_prime,
            out_channels=self.dod_in_channels,
            hidden_channels=self.dod_hidden_channels,
            latent_dim=self.n,
            num_layers=self.dod_num_layers,
            kernel=self.dod_kernel,
            stride=self.dod_stride,
            padding=self.dod_padding,
        )

    def make_pod_DFNN_kwargs(self) -> Dict[str, Any]:
        return dict(
            geometric_dim=self.parameter_mu_dim,
            physical_dim=self.parameter_nu_dim,
            n=self.n,
            layer_sizes=list(self.pod_df_layers),
        )

    def make_pod_Encoder_kwargs(self) -> Dict[str, Any]:
        return dict(
            input_dim=self.N,
            in_channels=self.pod_in_channels,
            hidden_channels=self.pod_hidden_channels,
            latent_dim=self.n,
            num_layers=self.pod_num_layers,
            kernel=self.pod_kernel,
            stride=self.pod_stride,
            padding=self.pod_padding,
        )

    def make_pod_Decoder_kwargs(self) -> Dict[str, Any]:
        return dict(
            output_dim=self.N,
            out_channels=self.pod_in_channels,
            hidden_channels=self.pod_hidden_channels,
            latent_dim=self.n,
            num_layers=self.pod_num_layers,
            kernel=self.pod_kernel,
            stride=self.pod_stride,
            padding=self.pod_padding,
        )

    def make_statDOD_kwargs(self) -> Dict[str, Any]:
        return dict(
            geometric_dim=self.parameter_mu_dim,
            seed_dim=self.preprocess_dim,
            num_roots=self.N_prime,
            root_output_dim=self.N,
            root_layer_sizes=list(self.stat_dod_structure),
        )

    def make_statHadamard_kwargs(self) -> Dict[str, Any]:
        return dict(
            param_mu_space_dim=self.parameter_mu_dim,
            param_nu_space_dim=self.parameter_nu_dim,
            m_0=self.stat_m,
            n_0=self.N_prime,
            layer_sizes=list(self.stat_phi_n_structure),
        )

    def make_CoLoRA_kwargs(self) -> Dict[str, Any]:
        return dict(
            out_dim=self.N,
            L=self.L,
            dyn_dim=self.N_prime,
            physical_dim=self.parameter_nu_dim,
            with_bias=True,
        )

    def trainer_defaults(self) -> Dict[str, Any]:
        return dict(
            epochs=self.generalepochs,
            restarts=self.generalrestarts,
            patience=self.generalpatience,
        )

    def dod_trainer_defaults(self) -> Dict[str, Any]:
        return dict(
            epochs=self.dod_epochs,
            restarts=self.dod_restarts,
            patience=self.dod_patience
        )



__all__ = ["Ex01Parameters", "Ex02Parameters", "Ex03Parameters"]
