import torch
from core import reduced_order_models as rom
from core.configs.parameters import Ex02Parameters

# --- Configure run / hyperparams --------------------------------------------
example_name = 'ex02'
P = Ex02Parameters(profile="baseline")                 # or "wide" / "tiny" / "debug"
P.assert_consistent()

trainer = P.trainer_defaults() 

# --- Data --------------------------------------------------------------------
train_valid_data      = rom.FetchTrainAndValidSet(0.8, example_name, 'N_A_reduced')
stat_train_valid_data = rom.FetchTrainAndValidSet(0.8, example_name, 'reduced_stationary')

# --- Stationary DOD ----------------------------------------------------------
stat_DOD_model = rom.statDOD(**P.make_statDOD_kwargs())

stat_DOD_Trainer = rom.statDODTrainer(
    stat_DOD_model, P.N_A,
    stat_train_valid_data,
    trainer["epochs"], trainer["restarts"],
    learning_rate=1e-3, batch_size=128, patience=trainer["patience"]
)
best_loss4 = stat_DOD_Trainer.train()

# --- Stationary coefficient model (HadamardNN) -------------------------------
stat_Coeff_model = rom.statHadamardNN(**P.make_statHadamard_kwargs())

stat_Coeff_Trainer = rom.statHadamardNNTrainer(
    stat_DOD_model, stat_Coeff_model, P.N_A,
    stat_train_valid_data,
    trainer["epochs"], trainer["restarts"],
    learning_rate=1e-3, batch_size=128, patience=trainer["patience"]
)
best_loss5 = stat_Coeff_Trainer.train()

# --- CoLoRA ------------------------------------------------------------------
CoLoRA_DL_model = rom.CoLoRA(**P.make_CoLoRA_kwargs())

CoLoRa_DL_Trainer = rom.CoLoRATrainer(
    P.Nt,
    stat_DOD_model, stat_Coeff_model, CoLoRA_DL_model,
    train_valid_data,
    trainer["epochs"], trainer["restarts"],
    learning_rate=1e-3, batch_size=128, patience=trainer["patience"]
)

best_loss6 = CoLoRa_DL_Trainer.train()
print(f"Best validation loss: {best_loss6}")

# --- Save modules ------------------------------------------------------------
torch.save(stat_DOD_model.state_dict(),   f'examples/{example_name}/state_dicts/stat_DOD_Module.pth')
torch.save(stat_Coeff_model.state_dict(), f'examples/{example_name}/state_dicts/stat_CoeffDOD_Module.pth')
torch.save(CoLoRA_DL_model.state_dict(),  f'examples/{example_name}/state_dicts/CoLoRA_Module.pth')
