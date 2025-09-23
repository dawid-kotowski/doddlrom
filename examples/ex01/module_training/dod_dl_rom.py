import torch
import numpy as np
from core import reduced_order_models as rom
from core.configs.parameters import Ex01Parameters

# --- Configure this run ------------------------------------------------------
example_name = 'ex01'
P = Ex01Parameters(profile="baseline")           
P.assert_consistent()

trainer_defaults = P.trainer_defaults()  

# --- Data --------------------------------------------------------------------
train_valid_data = rom.FetchTrainAndValidSet(0.8, example_name, 'N_reduced')

# --- DOD basis network -------------------------------------------------------
innerDOD_model = rom.innerDOD(**P.make_innerDOD_kwargs())
innerDOD_model.load_state_dict(torch.load(f'examples/{example_name}/state_dicts/DOD_Module.pth'))
innerDOD_model.eval()

# --- DOD-DL -----------------------------------------------------------------
En_model       = rom.Encoder(**P.make_dod_dl_Encoder_kwargs())
De_model       = rom.Decoder(**P.make_dod_dl_Decoder_kwargs())
DFNN_D_n_model = rom.DFNN(**P.make_dod_dl_DFNN_kwargs())

# --- Trainer -----------------------------------------------------------------
DFNN_D_n_trainer = rom.DOD_DL_ROMTrainer(
    P.Nt, 
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
    f'examples/{example_name}/state_dicts/DOD_DL_ROM_Module.pth'
)
