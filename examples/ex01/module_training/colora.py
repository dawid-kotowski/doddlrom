import torch
from master_project_1 import reduced_order_models as dr
from master_project_1.configs.ex01_parameters import Ex01Parameters

# --- Configure run / hyperparams --------------------------------------------
P = Ex01Parameters(profile="baseline")                 # or "wide" / "tiny" / "debug"
P.assert_consistent()

trainer = P.trainer_defaults() 

# --- Data --------------------------------------------------------------------
train_valid_data      = dr.FetchTrainAndValidSet(0.8, 'ex01', 'N_A_reduced')
stat_train_valid_data = dr.FetchTrainAndValidSet(0.8, 'ex01', 'reduced_stationary')

# --- Stationary DOD ----------------------------------------------------------
stat_DOD_model = dr.statDOD(**P.make_statDOD_kwargs())

stat_DOD_Trainer = dr.statDODTrainer(
    stat_DOD_model, P.N_A,
    stat_train_valid_data,
    trainer["epochs"], trainer["restarts"],
    learning_rate=1e-3, batch_size=128, patience=trainer["patience"]
)
best_loss4 = stat_DOD_Trainer.train()

# --- Stationary coefficient model (HadamardNN) -------------------------------
stat_Coeff_model = dr.statHadamardNN(**P.make_statHadamard_kwargs())

stat_Coeff_Trainer = dr.statHadamardNNTrainer(
    stat_DOD_model, stat_Coeff_model, P.N_A,
    stat_train_valid_data,
    trainer["epochs"], trainer["restarts"],
    learning_rate=1e-3, batch_size=128, patience=trainer["patience"]
)
best_loss5 = stat_Coeff_Trainer.train()

# --- CoLoRA ------------------------------------------------------------------
CoLoRA_DL_model = dr.CoLoRA(**P.make_CoLoRA_kwargs)

CoLoRa_DL_Trainer = dr.CoLoRATrainer(
    P.Nt,
    P.T,
    stat_DOD_model, stat_Coeff_model, CoLoRA_DL_model,
    train_valid_data,
    trainer["epochs"], trainer["restarts"],
    learning_rate=1e-3, batch_size=128, patience=trainer["patience"]
)

best_loss6 = CoLoRa_DL_Trainer.train()
print(f"Best validation loss: {best_loss6}")

# --- Save modules ------------------------------------------------------------
torch.save(stat_DOD_model.state_dict(),   'examples/ex01/state_dicts/stat_DOD_Module.pth')
torch.save(stat_Coeff_model.state_dict(), 'examples/ex01/state_dicts/stat_CoeffDOD_Module.pth')
torch.save(CoLoRA_DL_model.state_dict(),  'examples/ex01/state_dicts/CoLoRA_Module.pth')
