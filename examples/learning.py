"""
Learning script for {example_name}.

Run Programm:
-----------------------------------
python examples/learning.py 
--example "ex0{number}" [REQUIRED]
[OPTIONALS]
--epochs "epochs" 
--restarts "restarts" 
--with-stationary (default=FALSE)
--N "reduced dimension for POD-based"
--N-prime "reduced dimension for DOD-based"
--Ns "sample_size"
--Nt "sample_time_size"
-----------------------------------
"""
import argparse, importlib
from typing import Type
from pathlib import Path
import torch
from utils.paths import state_dicts_path
from core import reduced_order_models as rom


def load_parameters(example_name: str, profile: str):
    """
    Imports core/configs/parameters.py and returns a Parameters instance.
    The module should export Ex01Parameters / Ex02Parameters / Ex03Parameters, etc.
    """
    mod = importlib.import_module("core.configs.parameters")

    CLASS_BY_EXAMPLE = {
        "ex01": "Ex01Parameters",
        "ex02": "Ex02Parameters",
        "ex03": "Ex03Parameters",
    }
    try:
        cls_name = CLASS_BY_EXAMPLE[example_name]
    except KeyError:
        raise ImportError(f"Unsuitable Example Name: {example_name!r}")

    if not hasattr(mod, cls_name):
        raise ImportError(f"No class {cls_name} in core/configs/parameters.py")

    ParamCls: Type = getattr(mod, cls_name)
    return ParamCls(profile=profile)

def train_innerdod(P, example_name, trainer_overrides):
    # data: N_A-reduced
    tv = rom.FetchTrainAndValidSet(0.8, example_name, 'N_A_reduced')
    model = rom.innerDOD(**P.make_innerDOD_kwargs())                # (N_A -> N')
    trainer = rom.innerDODTrainer(P.Nt, model, tv,
                                  trainer_overrides["epochs"], trainer_overrides["restarts"],
                                  learning_rate=1e-3, batch_size=128, patience=trainer_overrides["patience"])
    best = trainer.train()
    torch.save(model.state_dict(), state_dicts_path(example_name) / f'DOD_Module.pth')
    return best

def train_doddfnn(P, example_name, trainer_overrides):
    # data: N_A-reduced
    tv = rom.FetchTrainAndValidSet(0.8, example_name, 'N_A_reduced')
    inner = rom.innerDOD(**P.make_innerDOD_kwargs())
    inner.load_state_dict(torch.load(state_dicts_path(example_name) / f'DOD_Module.pth', map_location='cpu'))
    rom._freeze(inner)
    coeff = rom.DFNN(**P.make_dod_dfnn_DFNN_kwargs())               # (p+q+1 -> N')
    trainer = rom.DFNNTrainer(P.Nt, P.N_A, inner, coeff, tv,
                              trainer_overrides["epochs"], trainer_overrides["restarts"],
                              learning_rate=1e-3, batch_size=128, patience=trainer_overrides["patience"])
    best = trainer.train()
    state_dict_path = state_dicts_path(example_name) / 'DODFNN_Module.pth'
    torch.save(coeff.state_dict(), state_dict_path)
    return best

def train_dod_dl_rom(P, example_name, trainer_overrides):
    # data: N_A-reduced
    tv = rom.FetchTrainAndValidSet(0.8, example_name, 'N_A_reduced')
    inner = rom.innerDOD(**P.make_innerDOD_kwargs())
    inner.load_state_dict(torch.load(state_dicts_path(example_name) / f'DOD_Module.pth', map_location='cpu'))
    rom._freeze(inner)
    enc = rom.Encoder(**P.make_dod_dl_Encoder_kwargs())             # (N' -> n)
    dec = rom.Decoder(**P.make_dod_dl_Decoder_kwargs())             # (n -> N')
    coeff = rom.DFNN(**P.make_dod_dl_DFNN_kwargs())                 # (p+q+1 -> n)
    trainer = rom.DOD_DL_ROMTrainer(P.Nt, inner, coeff, enc, dec, tv,
                                    0.6, trainer_overrides["epochs"], trainer_overrides["restarts"],
                                    learning_rate=1e-2, batch_size=128, patience=trainer_overrides["patience"])
    best = trainer.train()
    state_dict_path = state_dicts_path(example_name) / 'DOD_DL_ROM_Module.pth'
    torch.save({'encoder': enc.state_dict(), 'decoder': dec.state_dict(), 'coeff_model': coeff.state_dict()},
               state_dict_path)
    return best

def train_pod_dl_rom(P, example_name, trainer_overrides):
    # data: N-reduced
    tv = rom.FetchTrainAndValidSet(0.8, example_name, 'N_reduced')
    enc = rom.Encoder(**P.make_pod_Encoder_kwargs())                # (N -> n)
    dec = rom.Decoder(**P.make_pod_Decoder_kwargs())                # (n -> N)
    coeff = rom.DFNN(**P.make_pod_DFNN_kwargs())                    # (DFNN -> n)
    trainer = rom.POD_DL_ROMTrainer(P.Nt, coeff, enc, dec, tv,
                                    0.6, trainer_overrides["epochs"], trainer_overrides["restarts"],
                                    learning_rate=1e-2, batch_size=128, patience=trainer_overrides["patience"])
    best = trainer.train()
    state_dict_path = state_dicts_path(example_name) / 'POD_DL_ROM_Module.pth'
    torch.save({'encoder': enc.state_dict(), 'decoder': dec.state_dict(), 'coeff_model': coeff.state_dict()},
               state_dict_path)
    return best

def train_colora(P, example_name, trainer_overrides):
    # data: N-reduced + reduced_stationary
    tv_dyn = rom.FetchTrainAndValidSet(0.8, example_name, 'N_reduced')
    tv_stat = rom.FetchTrainAndValidSet(0.8, example_name, 'reduced_stationary')
    stat_dod = rom.statDOD(**P.make_statDOD_kwargs())               # (N' -> N_A)
    stat_dod_tr = rom.statDODTrainer(stat_dod, P.N, tv_stat,
                                     trainer_overrides["epochs"], trainer_overrides["restarts"],
                                     learning_rate=1e-3, batch_size=128, patience=trainer_overrides["patience"])
    _ = stat_dod_tr.train()
    rom._freeze(stat_dod)
    stat_coeff = rom.statHadamardNN(**P.make_statHadamard_kwargs()) # (p+q+1 -> N')
    stat_coeff_tr = rom.statHadamardNNTrainer(stat_dod, stat_coeff, P.N, tv_stat,
                                              trainer_overrides["epochs"], trainer_overrides["restarts"],
                                              learning_rate=1e-3, batch_size=128, patience=trainer_overrides["patience"])
    _ = stat_coeff_tr.train()
    rom._freeze(stat_coeff)
    colora = rom.CoLoRA(**P.make_CoLoRA_kwargs())                   # (p+q+1 -> N_A x Nt)
    colora_tr = rom.CoLoRATrainer(P.Nt, stat_dod, stat_coeff, colora, tv_dyn,
                                  trainer_overrides["epochs"], trainer_overrides["restarts"],
                                  learning_rate=1e-3, batch_size=128, patience=trainer_overrides["patience"])
    best = colora_tr.train()
    state_dict_path = state_dicts_path(example_name)
    torch.save(stat_dod.state_dict(),   state_dict_path / 'stat_DOD_Module.pth')
    torch.save(stat_coeff.state_dict(), state_dict_path / 'stat_CoeffDOD_Module.pth')
    torch.save(colora.state_dict(),     state_dict_path / 'CoLoRA_Module.pth')
    return best

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--example', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--restarts', type=int, default=None)
    parser.add_argument('--with-stationary', action='store_true', default=False)
    parser.add_argument('--N', type=int, default=None)
    parser.add_argument('--N-prime', type=int, default=None)
    parser.add_argument('--Ns', type=int, default=None)
    parser.add_argument('--Nt', type=int, default=None)
    args = parser.parse_args()

    P = load_parameters(args.example, profile='baseline')
    if args.N is not None: P.N = args.N
    if args.N_prime is not None: P.N_prime = args.N_prime
    if args.Ns is not None: P.Ns = args.Ns
    if args.Nt is not None: P.Nt = args.Nt
    P.assert_consistent()
    td = P.trainer_defaults()
    if args.epochs is not None:   td["epochs"]   = args.epochs
    if args.restarts is not None: td["restarts"] = args.restarts

    best_inner   = train_innerdod(P, args.example, td)
    best_doddfnn = train_doddfnn(P, args.example, td)
    best_doddl   = train_dod_dl_rom(P, args.example, td)
    best_poddl   = train_pod_dl_rom(P, args.example, td)

    result = {
        "best_innerDOD":   float(best_inner),
        "best_DOD_DFNN":   float(best_doddfnn),
        "best_DOD_DL_ROM": float(best_doddl),
        "best_POD_DL_ROM": float(best_poddl),
    }
    if args.with_stationary:
        result["best_CoLoRA"] = float(train_colora(P, args.example, td))

    print(result)



if __name__ == "__main__":
    main()