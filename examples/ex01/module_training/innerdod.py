import torch
from master_project_1 import reduced_order_models as dr
from master_project_1.configs.ex01_parameters import Ex01Parameters

# --- Configure run / hyperparams --------------------------------------------
P = Ex01Parameters(profile="baseline")                 # or "wide" / "tiny" / "debug"
P.assert_consistent()
trainer = P.trainer_defaults()  # {'epochs': ..., 'restarts': ..., 'patience': ...}

# --- Data --------------------------------------------------------------------
train_valid_data = dr.FetchTrainAndValidSet(0.8, 'ex01', 'N_A_reduced')

# --- DOD model ---------------------------------------------------------------
DOD_model = dr.innerDOD(**P.make_innerDOD_kwargs())

# --- Trainer -----------------------------------------------------------------
DOD_trainer = dr.innerDODTrainer(
    P.Nt,
    P.T,
    DOD_model,
    train_valid_data,
    trainer["epochs"],
    trainer["restarts"],
    learning_rate=1e-3,
    batch_size=128,
    patience=trainer["patience"],
)

best_loss = DOD_trainer.train()
print(f"Best validation loss: {best_loss}")

# --- Save --------------------------------------------------------------------
torch.save(DOD_model.state_dict(), 'examples/ex01/state_dicts/DOD_Module.pth')
