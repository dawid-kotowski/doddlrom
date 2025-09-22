"""
Learning script for {example_name}.

Run Programm:
-----------------------------------
python examples/learning.py 
--example "ex0{number}"
--profiles "profiles" 
--epochs "epochs" 
--restarts "restarts" 
-----------------------------------
"""
import argparse, importlib
from typing import Type
from pathlib import Path
import torch
from core import reduced_order_models as rom

# dims in comments below: Nt (time), N_A (ambient), N' (DOD cols), n (latent)

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

def ensure_dirs(example_name: str):
    sd = Path(f'examples/{example_name}/state_dicts')
    td = Path(f'examples/{example_name}/training_data')
    sd.mkdir(parents=True, exist_ok=True); td.mkdir(parents=True, exist_ok=True)

def train_innerdod(P, example_name, trainer_overrides):
    # data: N_A-reduced
    tv = rom.FetchTrainAndValidSet(0.8, example_name, 'N_A_reduced')
    model = rom.innerDOD(**P.make_innerDOD_kwargs())                # (N_A -> N')
    trainer = rom.innerDODTrainer(P.Nt, model, tv,
                                  trainer_overrides["epochs"], trainer_overrides["restarts"],
                                  learning_rate=1e-3, batch_size=128, patience=trainer_overrides["patience"])
    best = trainer.train()
    torch.save(model.state_dict(), f'examples/{example_name}/state_dicts/DOD_Module.pth')
    return best

def train_doddfnn(P, example_name, trainer_overrides):
    # data: N_A-reduced
    tv = rom.FetchTrainAndValidSet(0.8, example_name, 'N_A_reduced')
    inner = rom.innerDOD(**P.make_innerDOD_kwargs())
    inner.load_state_dict(torch.load(f'examples/{example_name}/state_dicts/DOD_Module.pth', map_location='cpu'))
    inner.eval()
    coeff = rom.DFNN(**P.make_dod_dfnn_DFNN_kwargs())               # (p+q+1 -> N')
    trainer = rom.DFNNTrainer(P.Nt, P.N_A, inner, coeff, tv,
                              trainer_overrides["epochs"], trainer_overrides["restarts"],
                              learning_rate=1e-3, batch_size=128, patience=trainer_overrides["patience"])
    best = trainer.train()
    torch.save(coeff.state_dict(), f'examples/{example_name}/state_dicts/DODFNN_Module.pth')
    return best

def train_dod_dl_rom(P, example_name, trainer_overrides):
    # data: N-reduced
    tv = rom.FetchTrainAndValidSet(0.8, example_name, 'N_reduced')
    inner = rom.innerDOD(**P.make_innerDOD_kwargs())
    inner.load_state_dict(torch.load(f'examples/{example_name}/state_dicts/DOD_Module.pth', map_location='cpu'))
    inner.eval()
    enc = rom.Encoder(**P.make_dod_dl_Encoder_kwargs())             # (N' -> n)
    dec = rom.Decoder(**P.make_dod_dl_Decoder_kwargs())             # (n -> N')
    coeff = rom.DFNN(**P.make_dod_dl_DFNN_kwargs())                 # (p+q+1 -> n)
    trainer = rom.DOD_DL_ROMTrainer(P.Nt, inner, coeff, enc, dec, tv,
                                    0.999, trainer_overrides["epochs"], trainer_overrides["restarts"],
                                    learning_rate=1e-3, batch_size=128, patience=trainer_overrides["patience"])
    best = trainer.train()
    torch.save({'encoder': enc.state_dict(), 'decoder': dec.state_dict(), 'coeff_model': coeff.state_dict()},
               f'examples/{example_name}/state_dicts/DOD_DL_ROM_Module.pth')
    return best

def train_pod_dl_rom(P, example_name, trainer_overrides):
    # data: N-reduced
    tv = rom.FetchTrainAndValidSet(0.8, example_name, 'N_reduced')
    enc = rom.Encoder(**P.make_pod_Encoder_kwargs())                # (N -> n)
    dec = rom.Decoder(**P.make_pod_Decoder_kwargs())                # (n -> N)
    coeff = rom.DFNN(**P.make_pod_DFNN_kwargs())                    # (DFNN -> n)
    trainer = rom.POD_DL_ROMTrainer(P.Nt, coeff, enc, dec, tv,
                                    0.999, trainer_overrides["epochs"], trainer_overrides["restarts"],
                                    learning_rate=1e-2, batch_size=128, patience=trainer_overrides["patience"])
    best = trainer.train()
    torch.save({'encoder': enc.state_dict(), 'decoder': dec.state_dict(), 'coeff_model': coeff.state_dict()},
               f'examples/{example_name}/state_dicts/POD_DL_ROM_Module.pth')
    return best

def train_colora(P, example_name, trainer_overrides):
    # data: N_A-reduced + reduced_stationary
    tv_dyn = rom.FetchTrainAndValidSet(0.8, example_name, 'N_A_reduced')
    tv_stat = rom.FetchTrainAndValidSet(0.8, example_name, 'reduced_stationary')
    stat_dod = rom.statDOD(**P.make_statDOD_kwargs())               # (N' -> N_A)
    stat_dod_tr = rom.statDODTrainer(stat_dod, P.N_A, tv_stat,
                                     trainer_overrides["epochs"], trainer_overrides["restarts"],
                                     learning_rate=1e-3, batch_size=128, patience=trainer_overrides["patience"])
    _ = stat_dod_tr.train()
    stat_coeff = rom.statHadamardNN(**P.make_statHadamard_kwargs()) # (p+q+1 -> N')
    stat_coeff_tr = rom.statHadamardNNTrainer(stat_dod, stat_coeff, P.N_A, tv_stat,
                                              trainer_overrides["epochs"], trainer_overrides["restarts"],
                                              learning_rate=1e-3, batch_size=128, patience=trainer_overrides["patience"])
    _ = stat_coeff_tr.train()
    colora = rom.CoLoRA(**P.make_CoLoRA_kwargs())                   # (p+q+1 -> N_A x Nt)
    colora_tr = rom.CoLoRATrainer(P.Nt, stat_dod, stat_coeff, colora, tv_dyn,
                                  trainer_overrides["epochs"], trainer_overrides["restarts"],
                                  learning_rate=1e-3, batch_size=128, patience=trainer_overrides["patience"])
    best = colora_tr.train()
    torch.save(stat_dod.state_dict(),   f'examples/{example_name}/state_dicts/stat_DOD_Module.pth')
    torch.save(stat_coeff.state_dict(), f'examples/{example_name}/state_dicts/stat_CoeffDOD_Module.pth')
    torch.save(colora.state_dict(),     f'examples/{example_name}/state_dicts/CoLoRA_Module.pth')
    return best

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--example', type=str, required=True)
    ap.add_argument('--profile', type=str, required=True)
    ap.add_argument('--epochs', type=int, default=None)
    ap.add_argument('--restarts', type=int, default=None)
    args = ap.parse_args()

    example_name = args.example
    ensure_dirs(example_name)

    P = load_parameters(example_name, profile=args.profile)
    P.assert_consistent()
    td = P.trainer_defaults()
    if args.epochs is not None:   td["epochs"] = args.epochs
    if args.restarts is not None: td["restarts"] = args.restarts

    # 1) innerDOD
    best_inner = train_innerdod(P, example_name, td)

    # 2) DOD+DFNN
    best_doddfnn = train_doddfnn(P, example_name, td)

    # 3) DOD-DL-ROM
    best_doddl = train_dod_dl_rom(P, example_name, td)

    # 4) POD-DL-ROM
    best_poddl = train_pod_dl_rom(P, example_name, td)

    # 5) CoLoRA
    best_colora = train_colora(P, example_name, td)

    print({"best_innerDOD": float(best_inner),
           "best_DOD_DFNN": float(best_doddfnn),
           "best_DOD_DL_ROM": float(best_doddl),
           "best_POD_DL_ROM": float(best_poddl),
           "best_CoLoRA": float(best_colora)})

if __name__ == "__main__":
    main()