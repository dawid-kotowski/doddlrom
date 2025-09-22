"""
Testing for {example_name}.

This is a runable program for the example01, that tests for different complexities
the respective ROMs.

Run Program
-----------------
python examples/{example_name}/test.py 
--profiles "profiles" 
--epochs "epochs" 
--restarts "restarts" 
--eval_samples "number of samples to test error with"
--outdir "path/to/benchmark/dir"
-----------------

"""
import argparse, json, time, csv, random, math
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm
from master_project_1 import reduced_order_models as rom
from master_project_1.configs.parameters import Ex01Parameters

def count_params(module, include_frozen=True):
    ps = list(module.parameters())
    if not include_frozen:
        ps = [p for p in ps if p.requires_grad]
    total = sum(p.numel() for p in ps)
    nonzero = sum((p != 0).sum().item() for p in ps)
    return total, nonzero

def freeze(module):
    for p in module.parameters():
        p.requires_grad = False
    module.eval()

def g_norm_batch(G, U):  # U: [Nt, Nh] or [Nh]
    if U.ndim == 1:
        return float(np.sqrt(U @ (G @ U) + 1e-12))
    vals = []
    for t in range(U.shape[0]):
        u = U[t]
        vals.append(float(np.sqrt(u @ (G @ u) + 1e-12)))
    return np.array(vals, dtype=np.float64)

def evaluate_rom_forward(rom_name, forward_fn, args, ref_sol, G):
    # forward_fn should return array with shape [Nt, Nh] (like eval.py forwards)
    U_hat = forward_fn(*args)
    Tm = min(ref_sol.shape[0], U_hat.shape[0])
    U_ref = ref_sol[:Tm]
    U_hat = U_hat[:Tm]
    diffs = U_ref - U_hat
    num = np.array([g_norm_batch(G, diffs[t]) for t in range(Tm)])
    den = np.array([g_norm_batch(G, U_ref[t])  for t in range(Tm)])
    abs_err = float(num.mean())
    rel_err = float((num / (den + 1e-12)).mean())
    return abs_err, rel_err

def time_one_forward(fn, *args, repeats=5):
    times = []
    for _ in range(repeats):
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        t0 = time.perf_counter()
        _ = fn(*args)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)
    return float(np.median(times))

def build_trainers_and_models(P, device, train_valid_set_N_A, train_valid_set_N):
    # inner DOD used by DOD+DFNN and DOD-DL-ROM
    innerDOD_model = rom.innerDOD(**P.make_innerDOD_kwargs()).to(device)
    inner_trainer = rom.innerDODTrainer(
        nt=P.Nt, dod_model=innerDOD_model,
        train_valid_set=train_valid_set_N_A,
        epochs=P.generalepochs,              
        restart=P.generalrestarts,
        learning_rate=1e-3,
        batch_size=128,
        device=device,
        patience=P.generalpatience,
    )

    # DOD+DFNN (DFNN -> N')
    dfnn_nprime = rom.DFNN(**P.make_dod_dfnn_DFNN_kwargs()).to(device)
    dfnn_trainer = rom.DFNNTrainer(
        nt=P.Nt, N_A=P.N_A, DOD_DL_model=innerDOD_model, coeffnn_model=dfnn_nprime,
        train_valid_set=train_valid_set_N_A, epochs=P.generalepochs, restarts=P.generalrestarts,
        learning_rate=1e-3, batch_size=128, device=device, patience=P.generalpatience
    )

    # DOD-DL-ROM (DFNN -> n, AE: N'<->n)
    coeff_n = rom.DFNN(**P.make_dod_dl_DFNN_kwargs()).to(device)
    enc = rom.Encoder(**P.make_dod_dl_Encoder_kwargs()).to(device)
    dec = rom.Decoder(**P.make_dod_dl_Decoder_kwargs()).to(device)
    doddl_trainer = rom.DOD_DL_ROMTrainer(
        nt=P.Nt, DOD_DL_model=innerDOD_model, Coeff_DOD_DL_model=coeff_n,
        Encoder_model=enc, Decoder_model=dec, train_valid_set=train_valid_set_N_A,
        error_weight=0.5, epochs=P.generalepochs, restarts=P.generalrestarts,
        learning_rate=1e-3, batch_size=128, device=device, patience=P.generalpatience
    )

    # POD-DL-ROM (DFNN -> n, AE: N_A<->n)
    pod_coeff = rom.DFNN(**P.make_pod_DFNN_kwargs()).to(device)
    pod_enc = rom.Encoder(**P.make_pod_Encoder_kwargs()).to(device)
    pod_dec = rom.Decoder(**P.make_pod_Decoder_kwargs()).to(device)
    pod_trainer = rom.POD_DL_ROMTrainer(
        nt=P.Nt, Coeff_model=pod_coeff, Encoder_model=pod_enc, Decoder_model=pod_dec,
        train_valid_set=train_valid_set_N, error_weight=0.5, epochs=P.generalepochs,
        restarts=P.generalrestarts, learning_rate=1e-3, batch_size=128, device=device,
        patience=P.generalpatience
    )

    models = {
        "DOD+DFNN": {"inner": innerDOD_model, "coeff": dfnn_nprime},
        "DOD-DL-ROM": {"inner": innerDOD_model, "coeff": coeff_n, "enc": enc, "dec": dec},
        "POD-DL-ROM": {"coeff": pod_coeff, "enc": pod_enc, "dec": pod_dec},
    }
    trainers = {
        "innerDOD": inner_trainer,
        "DOD+DFNN": dfnn_trainer,
        "DOD-DL-ROM": doddl_trainer,
        "POD-DL-ROM": pod_trainer,
    }
    return models, trainers

def forward_wrappers(P, device, models):
    A = np.load('examples/ex01/training_data/N_A_ambient_ex01.npz')['ambient'].astype(np.float32)
    A_P = np.load('examples/ex01/training_data/N_ambient_ex01.npz')['ambient'].astype(np.float32)
    A_t = torch.tensor(A, dtype=torch.float32, device=device)
    A_P_t = torch.tensor(A_P, dtype=torch.float32, device=device)

    def dod_dfnn(mu_i, nu_i):
        return rom.dod_dfnn_forward(A_t, models["DOD+DFNN"]["inner"], models["DOD+DFNN"]["coeff"],
                                    mu_i, nu_i, P.Nt, example_name='ex01', reduction_tag='N_A_reduced')

    def dod_dl(mu_i, nu_i):
        return rom.dod_dl_rom_forward(A_t, models["DOD-DL-ROM"]["inner"],
                                      models["DOD-DL-ROM"]["coeff"], models["DOD-DL-ROM"]["dec"],
                                      mu_i, nu_i, P.Nt, example_name='ex01', reduction_tag='N_A_reduced')

    def pod_dl(mu_i, nu_i):
        return rom.pod_dl_rom_forward(A_P_t, models["POD-DL-ROM"]["coeff"],
                                      models["POD-DL-ROM"]["dec"], mu_i, nu_i, 
                                      P.Nt, example_name='ex01', reduction_tag='N_reduced')

    return {"DOD+DFNN": dod_dfnn, "DOD-DL-ROM": dod_dl, "POD-DL-ROM": pod_dl}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--profiles', nargs='+', default=['baseline','wide','tiny','debug'])
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--restarts', type=int, default=None)
    parser.add_argument('--eval_samples', type=int, default=5)
    parser.add_argument('--outdir', type=str, default='examples/ex01/benchmarks')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    # Data
    tv_NA = rom.FetchTrainAndValidSet(0.8, 'ex01', 'N_A_reduced')
    tv_N = rom.FetchTrainAndValidSet(0.8, 'ex01', 'N_reduced')
    full = np.load('examples/ex01/training_data/full_order_training_data_ex01.npz')
    mu_full, nu_full, sol_full = full['mu'], full['nu'], full['solution']

    G = np.load('examples/ex01/training_data/gram_matrix_ex01.npz')['gram'].astype(np.float32)

    # metrics file
    csv_path = outdir / 'rom_sweep.csv'
    first_write = not csv_path.exists()
    with open(csv_path, 'a', newline='') as fcsv:
        writer = csv.DictWriter(fcsv, fieldnames=[
            'profile','rom','epochs','restarts','val_loss','params_total','params_nonzero',
            'forward_ms','abs_L2G','rel_L2G'
        ])
        if first_write: writer.writeheader()

        for profile in args.profiles:
            P = Ex01Parameters(profile=profile)
            P.assert_consistent()
            if args.epochs is not None: P.generalepochs = args.epochs
            if args.restarts is not None: P.generalrestarts = args.restarts

            models, trainers = build_trainers_and_models(P, device, tv_NA, tv_N)
            fw = forward_wrappers(P, device, models)

            # Train per ROM
            val_losses = {}
            for rom_name, trainer in trainers.items():
                val_losses[rom_name] = trainer.train()
                if rom_name == "innerDOD":
                    freeze(models["DOD+DFNN"]["inner"])
                    freeze(models["DOD-DL-ROM"]["inner"])


            # Pick random eval items
            Ns_1 = mu_full.shape[0]
            idxs = np.random.choice(np.arange(Ns_1), size=min(args.eval_samples, Ns_1), replace=False)

            for rom_name in ["DOD+DFNN","DOD-DL-ROM","POD-DL-ROM"]:
                # Count params
                if rom_name == "DOD+DFNN":
                    modules = [models[rom_name]["inner"], models[rom_name]["coeff"]]
                elif rom_name == "DOD-DL-ROM":
                    modules = [models[rom_name]["inner"], models[rom_name]["coeff"], models[rom_name]["enc"], models[rom_name]["dec"]]
                else:
                    modules = [models[rom_name]["coeff"], models[rom_name]["enc"], models[rom_name]["dec"]]
                total = nonzero = 0
                for m in modules:
                    t, nz = count_params(m, include_frozen=True)
                    total += t; nonzero += nz

                # Errors and forward time: use first selected sample for timing
                i0 = int(idxs[0])
                mu_i = torch.tensor([[float(mu_full[i0])]], dtype=torch.float32, device=device)
                nu_i = torch.tensor([[float(nu_full[i0])]], dtype=torch.float32, device=device)

                # forward timing
                fwd_ms = time_one_forward(fw[rom_name], mu_i, nu_i, repeats=5)

                # errors (averaged across chosen samples)
                abs_list, rel_list = [], []
                for i in idxs:
                    mu_j = torch.tensor([[float(mu_full[int(i)])]], dtype=torch.float32, device=device)
                    nu_j = torch.tensor([[float(nu_full[int(i)])]], dtype=torch.float32, device=device)
                    ref = sol_full[int(i)]  # [Nt, Nh]
                    abs_e, rel_e = evaluate_rom_forward(
                        rom_name, fw[rom_name], (mu_j, nu_j), ref, G
                    )
                    abs_list.append(abs_e); rel_list.append(rel_e)

                row = {
                    'profile': profile,
                    'rom': rom_name,
                    'epochs': P.generalepochs,
                    'restarts': P.generalrestarts,
                    'val_loss': float(val_losses[rom_name]),
                    'params_total': int(total),
                    'params_nonzero': int(nonzero),
                    'forward_ms': float(fwd_ms),
                    'abs_L2G': float(np.mean(abs_list)),
                    'rel_L2G': float(np.mean(rel_list)),
                }
                writer.writerow(row)
                print(json.dumps(row))

if __name__ == "__main__":
    main()