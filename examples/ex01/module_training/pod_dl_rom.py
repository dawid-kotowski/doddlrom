import torch
from core import reduced_order_models as dr
from core.configs.parameters import Ex01Parameters
from utils.paths import state_dicts_path

# --- Configure / hyperparams -------------------------------------------------
example_name = 'ex01'
P = Ex01Parameters(profile="baseline")     
P.assert_consistent()

trainer = P.trainer_defaults()  

# --- Data --------------------------------------------------------------------
train_valid_data = dr.FetchTrainAndValidSet(0.8, example_name, 'N_reduced')

# --- POD-DL components (DFNN -> n, AE maps N_A <-> n) -----------------------
En_model       = dr.Encoder(**P.make_pod_Encoder_kwargs())
De_model       = dr.Decoder(**P.make_pod_Decoder_kwargs())
DFNN_P_n_model = dr.DFNN(**P.make_pod_DFNN_kwargs())


# --- Trainer -----------------------------------------------------------------
DFNN_P_n_trainer = dr.POD_DL_ROMTrainer(
    P.Nt, 
    DFNN_P_n_model, En_model, De_model,
    train_valid_data, 0.999,
    trainer["epochs"], trainer["restarts"],
    learning_rate=1e-2, batch_size=128, patience=trainer["patience"]
)

best_loss3 = DFNN_P_n_trainer.train()
print(f"Best validation loss: {best_loss3}")

# --- Save --------------------------------------------------------------------
torch.save(
    {
        'encoder': En_model.state_dict(),
        'decoder': De_model.state_dict(),
        'coeff_model': DFNN_P_n_model.state_dict(),
    },
    state_dicts_path(example_name) / f'POD_DL_ROM_Module.pth'
)
