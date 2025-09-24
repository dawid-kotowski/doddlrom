import torch
from core import reduced_order_models as rom
from core.configs.parameters import Ex03Parameters

# --- Configure / hyperparams -------------------------------------------------
example_name = 'ex03'
P = Ex03Parameters(profile="baseline")        
P.assert_consistent()

trainer = P.trainer_defaults()  

# --- Data --------------------------------------------------------------------
train_valid_data = rom.FetchTrainAndValidSet(0.8, example_name, 'N_A_reduced')

# --- DOD model (pretrained basis) -------------------------------------------
innerDOD_model = rom.innerDOD(**P.make_innerDOD_kwargs())
innerDOD_model.load_state_dict(torch.load(f'examples/{example_name}/state_dicts/DOD_Module.pth'))
innerDOD_model.eval()

# --- DFNN  ------------------------------------------------------------------
DOD_coeff_model = rom.DFNN(**P.make_dod_dfnn_DFNN_kwargs())

# --- Trainer -----------------------------------------------------------------
DOD_coeff_trainer = rom.DFNNTrainer(
    P.Nt,
    P.N_A,                           
    innerDOD_model,
    DOD_coeff_model,
    train_valid_data,
    trainer["epochs"],
    trainer["restarts"],
    learning_rate=1e-3,
    batch_size=128,
    patience=trainer["patience"],
)

best_loss = DOD_coeff_trainer.train()
print(f"Best validation loss: {best_loss}")

# --- Save --------------------------------------------------------------------
torch.save(DOD_coeff_model.state_dict(), f'examples/{example_name}/state_dicts/DODFNN_Module.pth')
