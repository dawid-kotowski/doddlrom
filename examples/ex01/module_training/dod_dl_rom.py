import torch
import numpy as np
from master_project_1 import reduced_order_models as dr
from master_project_1.configs.ex01_parameters import Ex01Parameters

# --- Configure this run ------------------------------------------------------
P = Ex01Parameters(profile="baseline")            # or "wide" / "tiny" / "debug"
P.assert_consistent()

trainer_defaults = P.trainer_defaults()  

# --- Data --------------------------------------------------------------------
train_valid_data = dr.FetchTrainAndValidSet(0.8, 'ex01', 'N_reduced')

# --- DOD basis network -------------------------------------------------------
innerDOD_model = dr.innerDOD(**P.make_innerDOD_kwargs())
innerDOD_model.load_state_dict(torch.load('examples/ex01/state_dicts/DOD_Module.pth'))
innerDOD_model.eval()

# --- DOD-DL -----------------------------------------------------------------
En_model       = dr.Encoder(**P.make_dod_dl_Encoder_kwargs())
De_model       = dr.Decoder(**P.make_dod_dl_Decoder_kwargs())
DFNN_D_n_model = dr.DFNN(**P.make_dod_dl_DFNN_kwargs())

# --- Trainer -----------------------------------------------------------------
DFNN_D_n_trainer = dr.DOD_DL_ROMTrainer(
    P.Nt,
    P.T,
    innerDOD_model,
    DFNN_D_n_model,
    En_model,
    De_model,
    train_valid_data,
    0.999,
    trainer_defaults["epochs"],
    trainer_defaults["restarts"],
    learning_rate=1e-3,
    batch_size=128,
    patience=trainer_defaults["patience"],
)

best_loss = DFNN_D_n_trainer.train()
print(f"Best validation loss: {best_loss}")

# --- Save modules ------------------------------------------------------------
torch.save(
    {
        'encoder': En_model.state_dict(),
        'decoder': De_model.state_dict(),
        'coeff_model': DFNN_D_n_model.state_dict(),
    },
    'examples/ex01/state_dicts/DOD_DL_ROM_Module.pth'
)
