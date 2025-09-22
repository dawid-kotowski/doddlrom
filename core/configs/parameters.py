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

# ---------------- Presets for layer shapes (problem sizes stay fixed) --------
_PRESETS: Dict[str, Dict[str, Any]] = {
    "baseline": {
        # DOD only
        "preprocess_dim": 2,           
        "dod_structure": [64],     

        # DOD+DFNN 
        "df_layers": [16, 8],          

        # DOD-DL-ROM 
        "dod_dl_df_layers": [16, 8],
        "dod_in_channels": 1,
        "dod_hidden_channels": 1,
        "dod_lin_dim_ae": 0,
        "dod_kernel": 3,
        "dod_stride": 2,
        "dod_padding": 1,
        "dod_num_layers": 1,           

        # POD-DL-ROM 
        "pod_df_layers": [16, 8],
        "pod_in_channels": 1,
        "pod_hidden_channels": 1,
        "pod_lin_dim_ae": 0,
        "pod_kernel": 3,
        "pod_stride": 2,
        "pod_padding": 1,
        "pod_num_layers": 2,

        # Stationary + CoLoRA
        "L": 3,
        "stat_m": 4,                    
        "stat_dod_structure": [128, 64],
        "stat_phi_n_structure": [16, 8],

        # Training defaults
        "generalepochs": 500,
        "generalrestarts": 2,
        "generalpatience": 3,
    },
    "wide": {
        "dod_structure": [128, 64],
        "df_layers": [32, 16],
        "dod_dl_df_layers": [32, 16],
        "pod_df_layers": [32, 16],
        "stat_dod_structure": [128, 64],
        "stat_phi_n_structure": [32, 16],
        "dod_num_layers": 1,
        "pod_num_layers": 3,
    },
    "deep": {
        "dod_structure": [64, 64, 64],
        "df_layers": [16, 16, 16],
        "dod_dl_df_layers": [16, 16, 16],
        "pod_df_layers": [16, 16, 16],
        "stat_dod_structure": [64, 64, 64],
        "stat_phi_n_structure": [16, 16, 16],
        "dod_num_layers": 1,
        "pod_num_layers": 3,
    },
    "tiny": {
        "dod_structure": [32],
        "df_layers": [8],
        "dod_dl_df_layers": [8],
        "pod_df_layers": [8],
        "stat_dod_structure": [32],
        "stat_phi_n_structure": [8],
        "dod_num_layers": 1,
        "pod_num_layers": 3,
    },
    "debug": {
        "dod_structure": [8],
        "df_layers": [4],
        "dod_dl_df_layers": [8, 4],
        "pod_df_layers": [8, 4],
        "stat_dod_structure": [8],
        "stat_phi_n_structure": [4],
        "dod_num_layers": 1,
        "pod_num_layers": 3,
    },
}


@dataclass
class Ex01Parameters:
    # -------- Fixed example sizes (problem-level) ----------------------------
    N_h: int = 5101
    N_A: int = 64
    N: int = 64
    N_prime: int = 16
    n: int = 4
    Nt: int = 10
    Ns: int = 400
    T: float = 1.
    diameter: float = 0.02
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
    dod_lin_dim_ae: int = 0
    dod_kernel: int = 3
    dod_stride: int = 2
    dod_padding: int = 1
    dod_num_layers: int = 2

    # -------- POD-DL-ROM (AE on N_A <-> n) ----------------------------------
    pod_df_layers: List[int] = field(default_factory=lambda: [32, 16, 8])
    pod_in_channels: int = 1
    pod_hidden_channels: int = 1
    pod_lin_dim_ae: int = 0
    pod_kernel: int = 3
    pod_stride: int = 2
    pod_padding: int = 1
    pod_num_layers: int = 2

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
        preset = _PRESETS.get(self.profile, {})
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
            root_output_dim=self.N_A,
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
            out_dim=self.N_A,
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


@dataclass
class Ex02Parameters:
    # -------- Fixed example sizes (problem-level) ----------------------------
    N_h: int = 222211
    N_A: int = 64
    N: int = 64
    N_prime: int = 16
    n: int = 4
    Nt: int = 10
    Ns: int = 16
    T: float = 1.
    diameter: float = 0.01
    parameter_mu_dim: int = 3
    parameter_nu_dim: int = 1

    # -------- innerDOD ------------------------------------------------------
    preprocess_dim: int = 4
    dod_structure: List[int] = field(default_factory=lambda: [32, 16])

    # -------- DOD+DFNN ------------------------------------------------------
    df_layers: List[int] = field(default_factory=lambda: [16, 8])

    # -------- DOD-DL-ROM (AE on N' <-> n) -----------------------------------
    dod_dl_df_layers: List[int] = field(default_factory=lambda: [32, 16, 8])
    dod_in_channels: int = 1
    dod_hidden_channels: int = 1
    dod_lin_dim_ae: int = 0
    dod_kernel: int = 3
    dod_stride: int = 2
    dod_padding: int = 1
    dod_num_layers: int = 2

    # -------- POD-DL-ROM (AE on N_A <-> n) ----------------------------------
    pod_df_layers: List[int] = field(default_factory=lambda: [32, 16, 8])
    pod_in_channels: int = 1
    pod_hidden_channels: int = 1
    pod_lin_dim_ae: int = 0
    pod_kernel: int = 3
    pod_stride: int = 2
    pod_padding: int = 1
    pod_num_layers: int = 2

    # -------- Stationary + CoLoRA ------------------------------------------
    L: int = 3
    stat_m: int = 4
    stat_dod_structure: List[int] = field(default_factory=lambda: [128, 64])
    stat_phi_n_structure: List[int] = field(default_factory=lambda: [16, 8])

    # -------- Training defaults ---------------------------------------------
    generalepochs: int = 500
    generalrestarts: int = 3
    generalpatience: int = 2

    # -------- Meta -----------------------------------------------------------
    profile: str = "baseline"

    def __post_init__(self) -> None:
        preset = _PRESETS.get(self.profile, {})
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
            root_output_dim=self.N_A,
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
            out_dim=self.N_A,
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


@dataclass
class Ex03Parameters:
    # -------- Fixed example sizes (problem-level) ----------------------------
    N_h: int = 5101
    N_A: int = 64
    N: int = 64
    N_prime: int = 16
    n: int = 4
    Nt: int = 10
    Ns: int = 400
    T: float = 1.
    diameter: float = 0.02
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
    dod_lin_dim_ae: int = 0
    dod_kernel: int = 3
    dod_stride: int = 2
    dod_padding: int = 1
    dod_num_layers: int = 2

    # -------- POD-DL-ROM (AE on N_A <-> n) ----------------------------------
    pod_df_layers: List[int] = field(default_factory=lambda: [32, 16, 8])
    pod_in_channels: int = 1
    pod_hidden_channels: int = 1
    pod_lin_dim_ae: int = 0
    pod_kernel: int = 3
    pod_stride: int = 2
    pod_padding: int = 1
    pod_num_layers: int = 2

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
        preset = _PRESETS.get(self.profile, {})
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
            root_output_dim=self.N_A,
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
            out_dim=self.N_A,
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
