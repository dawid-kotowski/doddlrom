"""
Parameters / hyperparameter registry for examples/.

This container is tailored to current constructors & trainers in
`reduced_order_models.py` (innerDOD, DFNN, HadamardNN, Encoder/Decoder,
statDOD, statHadamardNN, CoLoRA, and all *Trainer classes*).

Key goals
- Centralize all fixed problem sizes (N_h, N_A, N, N_prime, n, Nt, ...)
- Offer switchable presets that only change network layers (problem sizes unchanged)
- Provide *factory-style* kwargs that match constructors exactly

======================================
Example Build using Parameter registry
======================================
Import
------
>>> from examples.ex01.parameters import Ex01Parameters
>>> P = Ex01Parameters(profile="baseline")  # or "wide", "tiny", "debug"
-------
# Build innerDOD
innerDOD_kwargs = P.make_innerDOD_kwargs()
inner_dod = innerDOD(**innerDOD_kwargs)

# Build DOD+DFNN (HadamardNN or DFNN that outputs N')
dod_dfnn_kwargs = P.make_dod_dfnn_DFNN_kwargs()          # DFNN -> N'
dod_hadamard_kwargs = P.make_dod_dfnn_Hadamard_kwargs()  # HadamardNN -> N'

# Build DOD-DL-ROM (Coeff DFNN -> n, AE: N' <-> n)
coeff_kwargs = P.make_dod_dl_DFNN_kwargs()           # DFNN -> n
enc_kwargs    = P.make_dod_dl_Encoder_kwargs()       # Encoder: N' -> n
dec_kwargs    = P.make_dod_dl_Decoder_kwargs()       # Decoder: n  -> N'

# Build POD-DL-ROM (Coeff DFNN -> n, AE: N <-> n)
pod_coeff_kwargs = P.make_pod_DFNN_kwargs()          # DFNN -> n
pod_enc_kwargs   = P.make_pod_Encoder_kwargs()       # Encoder: N -> n
pod_dec_kwargs   = P.make_pod_Decoder_kwargs()       # Decoder: n   -> N

# Build stationary + CoLoRA
stat_dod_kwargs      = P.make_statDOD_kwargs()
stat_hadamard_kwargs = P.make_statHadamard_kwargs()  # -> n
colora_kwargs        = P.make_CoLoRA_kwargs()        # (out_dim=N_A)
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict, replace
from pathlib import Path
from typing import Any, Dict, List
import json


EX01_PRESETS: Dict[str, Dict[str, Any]] = {
    "baseline": {
        # DOD only
        "preprocess_dim": 2,           
        "dod_structure": [128, 64],     

        # DOD+DFNN 
        "df_layers": [16, 8],          

        # DOD-DL-ROM 
        "dod_dl_df_layers": [16, 8],  
        "dod_in_channels": 1,
        "dod_hidden_channels": 1,  
        "dod_kernel": 3,
        "dod_stride": 2,
        "dod_padding": 1,
        "dod_num_layers": 1,

        # POD-DL-ROM 
        "pod_df_layers": [16, 8],
        "pod_in_channels": 1,
        "pod_hidden_channels": 1,
        "pod_kernel": 3,
        "pod_stride": 2,
        "pod_padding": 1,
        "pod_num_layers": 2,

        # Stationary + CoLoRA
        "L": 3,                 
        "stat_dod_structure": [128, 64],
        "stat_phi_n_structure": [16, 8],
    },
    "test1": {
        "preprocess_dim": 2,           
        "dod_structure": [32],
        "df_layers": [8],
        "dod_dl_df_layers": [8],  
        "dod_in_channels": 1,
        "dod_hidden_channels": 2,
        "dod_kernel": 3,
        "dod_stride": 2,
        "dod_padding": 1,
        "dod_num_layers": 1,
        "pod_df_layers": [8],
        "pod_in_channels": 1,
        "pod_hidden_channels": 2, 
        "pod_kernel": 3,
        "pod_stride": 2,
        "pod_padding": 1,
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
}

@dataclass
class Ex01Parameters:
    # -------- Fixed example sizes (problem-level) ----------------------------
    N_A: int = 32
    N: int = 16
    N_prime: int = 9
    n: int = 2
    Nt: int = 30
    Ns: int = 400
    T: float = 1.
    diameter: float = 0.03
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
    dod_hidden_channels: int = 1
    dod_kernel: int = 3
    dod_stride: int = 2
    dod_padding: int = 1
    dod_num_layers: int = 1

    # -------- POD-DL-ROM (AE on N_A <-> n) ----------------------------------
    pod_df_layers: List[int] = field(default_factory=lambda: [32, 16, 8])
    pod_in_channels: int = 1
    pod_hidden_channels: int = 1
    pod_kernel: int = 3
    pod_stride: int = 2
    pod_padding: int = 1
    pod_num_layers: int = 3

    # -------- Stationary + CoLoRA ------------------------------------------
    L: int = 3
    stat_m: int = 4
    stat_dod_structure: List[int] = field(default_factory=lambda: [128, 64])
    stat_phi_n_structure: List[int] = field(default_factory=lambda: [16, 8])

    # -------- Training defaults ---------------------------------------------
    generalepochs: int = 20
    generalrestarts: int = 3
    generalpatience: int = 3

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


EX02_PRESETS: Dict[str, Dict[str, Any]] = {
    "baseline": {
        "preprocess_dim": 3,
        "dod_structure": [128, 64],

        "df_layers": [32, 16],

        # DOD-DL-ROM (AE on N'↔n; 4×4 grid)
        "dod_dl_df_layers": [32, 16],
        "dod_in_channels": 1,
        "dod_hidden_channels": 8,
        "dod_kernel": 5, "dod_stride": 2, "dod_padding": 2,
        "dod_num_layers": 2,

        # POD-DL-ROM (AE; see note above re N_A vs N)
        "pod_df_layers": [32, 16],
        "pod_in_channels": 1,
        "pod_hidden_channels": 8,
        "pod_kernel": 5, "pod_stride": 2, "pod_padding": 2,
        "pod_num_layers": 4,

        "L": 3,
        "stat_dod_structure": [128, 64],
        "stat_phi_n_structure": [32, 16],
    },

    "test1": {  
        "preprocess_dim": 2,
        "dod_structure": [32],

        "df_layers": [8],

        "dod_dl_df_layers": [8],
        "dod_in_channels": 1,
        "dod_hidden_channels": 4,
        "dod_kernel": 3, "dod_stride": 2, "dod_padding": 1,
        "dod_num_layers": 1,

        "pod_df_layers": [8],
        "pod_in_channels": 1,
        "pod_hidden_channels": 4,
        "pod_kernel": 3, "pod_stride": 2, "pod_padding": 1,
        "pod_num_layers": 3,

        "L": 2,
        "stat_dod_structure": [32],
        "stat_phi_n_structure": [8],
    },

    "test2": {  
        "preprocess_dim": 3,
        "dod_structure": [64, 32],

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
        "pod_num_layers": 4,

        "L": 3,
        "stat_dod_structure": [64, 32],
        "stat_phi_n_structure": [16, 8],
    },

    "test3": {  
        "preprocess_dim": 3,
        "dod_structure": [128, 64],

        "df_layers": [32, 16],

        "dod_dl_df_layers": [32, 16],
        "dod_in_channels": 1,
        "dod_hidden_channels": 12,
        "dod_kernel": 5, "dod_stride": 2, "dod_padding": 2,
        "dod_num_layers": 2,

        "pod_df_layers": [32, 16],
        "pod_in_channels": 1,
        "pod_hidden_channels": 12,
        "pod_kernel": 5, "pod_stride": 2, "pod_padding": 2,
        "pod_num_layers": 5,

        "L": 3,
        "stat_dod_structure": [128, 64],
        "stat_phi_n_structure": [32, 16],
    },

    "test4": { 
        "preprocess_dim": 4,
        "dod_structure": [128, 64, 64],

        "df_layers": [48, 32, 16],

        "dod_dl_df_layers": [48, 32, 16],
        "dod_in_channels": 1,
        "dod_hidden_channels": 16,
        "dod_kernel": 5, "dod_stride": 2, "dod_padding": 2,
        "dod_num_layers": 3,

        "pod_df_layers": [48, 32, 16],
        "pod_in_channels": 1,
        "pod_hidden_channels": 16,
        "pod_kernel": 5, "pod_stride": 2, "pod_padding": 2,
        "pod_num_layers": 5,

        "L": 4,
        "stat_dod_structure": [128, 64, 64],
        "stat_phi_n_structure": [48, 32, 16],
    },

    "test5": {  
        "preprocess_dim": 4,
        "dod_structure": [256, 128, 64],

        "df_layers": [64, 48, 24],

        "dod_dl_df_layers": [64, 48, 24],
        "dod_in_channels": 1,
        "dod_hidden_channels": 24,
        "dod_kernel": 5, "dod_stride": 2, "dod_padding": 2,
        "dod_num_layers": 3,

        "pod_df_layers": [64, 48, 24],
        "pod_in_channels": 1,
        "pod_hidden_channels": 24,
        "pod_kernel": 5, "pod_stride": 2, "pod_padding": 2,
        "pod_num_layers": 5,

        "L": 4,
        "stat_dod_structure": [256, 128, 64],
        "stat_phi_n_structure": [64, 48, 24],
    },
}



@dataclass
class Ex02Parameters:
    # -------- Fixed example sizes (problem-level) ----------------------------
    # Note: to achieve l2_err=1e-5, N_A must be 478
    N_A: int = 478
    N: int = 256
    N_prime: int = 16
    n: int = 8
    Nt: int = 10
    Ns: int = 400
    T: float = 1.
    diameter: float = 0.02
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
    dod_hidden_channels: int = 1
    dod_lin_dim_ae: int = 0
    dod_kernel: int = 3
    dod_stride: int = 2
    dod_padding: int = 1
    dod_num_layers: int = 1

    # -------- POD-DL-ROM (AE on N <-> n + DFNN on n <-> p+q+1) -------------
    pod_df_layers: List[int] = field(default_factory=lambda: [32, 16, 8])
    pod_in_channels: int = 1
    pod_hidden_channels: int = 1
    pod_lin_dim_ae: int = 0
    pod_kernel: int = 3
    pod_stride: int = 2
    pod_padding: int = 1
    pod_num_layers: int = 3

    # -------- Stationary + CoLoRA ------------------------------------------
    L: int = 3
    stat_m: int = 8
    stat_dod_structure: List[int] = field(default_factory=lambda: [128, 64])
    stat_phi_n_structure: List[int] = field(default_factory=lambda: [16, 8])

    # -------- Training defaults ---------------------------------------------
    generalepochs: int = 5000
    generalrestarts: int = 10
    generalpatience: int = 50

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


EX03_PRESETS: Dict[str, Dict[str, Any]] = {

    "baseline": {
        "preprocess_dim": 2,
        "dod_structure": [100],

        "df_layers": [16, 8],

        # DOD-DL-ROM (AE on N'=9 ↔ n=2; 3x3 grid)
        "dod_dl_df_layers": [16, 8],
        "dod_in_channels": 1,
        "dod_hidden_channels": 1,
        "dod_kernel": 3, "dod_stride": 2, "dod_padding": 1,
        "dod_num_layers": 2,      # 3→2→1

        # POD-DL-ROM (AE on N=64 ↔ n=2; 8x8 grid)
        "pod_df_layers": [16, 8],
        "pod_in_channels": 1,
        "pod_hidden_channels": 1,
        "pod_kernel": 3, "pod_stride": 2, "pod_padding": 1,
        "pod_num_layers": 3,      # 8→4→2→1

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
        "dod_num_layers": 1,      # 3→2
        "pod_df_layers": [8],
        "pod_in_channels": 1,
        "pod_hidden_channels": 2,
        "pod_kernel": 3, "pod_stride": 2, "pod_padding": 1,
        "pod_num_layers": 2,      # 8→4→2
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
        "dod_num_layers": 2,      # 3→2→1
        "pod_df_layers": [12, 8],
        "pod_in_channels": 1,
        "pod_hidden_channels": 4,
        "pod_kernel": 3, "pod_stride": 2, "pod_padding": 1,
        "pod_num_layers": 3,      # 8→4→2→1
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
        "dod_num_layers": 2,      # 3→2→1
        "pod_df_layers": [16, 12],
        "pod_in_channels": 1,
        "pod_hidden_channels": 8,
        "pod_kernel": 5, "pod_stride": 2, "pod_padding": 2,
        "pod_num_layers": 3,      # 8→4→2→1
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
        "dod_num_layers": 3,      # 3→2→1→1
        "pod_df_layers": [24, 16, 8],
        "pod_in_channels": 1,
        "pod_hidden_channels": 12,
        "pod_kernel": 5, "pod_stride": 2, "pod_padding": 2,
        "pod_num_layers": 4,      # 8→4→2→1→1
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
}


@dataclass
class Ex03Parameters:
    # -------- Fixed example sizes (problem-level) ----------------------------
    N_A: int = 101
    N: int = 64
    N_prime: int = 9
    n: int = 2
    Nt: int = 30
    Ns: int = 400
    T: float = 3.
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
    dod_hidden_channels: int = 1
    dod_lin_dim_ae: int = 0
    dod_kernel: int = 3
    dod_stride: int = 2
    dod_padding: int = 1
    dod_num_layers: int = 1

    # -------- POD-DL-ROM (AE on N <-> n + DFNN on n <-> p+q+1) -------------
    pod_df_layers: List[int] = field(default_factory=lambda: [32, 16, 8])
    pod_in_channels: int = 1
    pod_hidden_channels: int = 1
    pod_lin_dim_ae: int = 0
    pod_kernel: int = 3
    pod_stride: int = 2
    pod_padding: int = 1
    pod_num_layers: int = 3

    # -------- Stationary + CoLoRA ------------------------------------------
    L: int = 3
    stat_m: int = 8
    stat_dod_structure: List[int] = field(default_factory=lambda: [128, 64])
    stat_phi_n_structure: List[int] = field(default_factory=lambda: [16, 8])

    # -------- Training defaults ---------------------------------------------
    generalepochs: int = 5000
    generalrestarts: int = 5
    generalpatience: int = 50

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









__all__ = ["Ex01Parameters", "Ex02Parameters", "Ex03Parameters"]
