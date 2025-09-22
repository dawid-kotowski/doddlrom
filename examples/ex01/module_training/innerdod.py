import torch
from core import reduced_order_models as rom
from core.configs.parameters import Ex01Parameters

# --- Configure run / hyperparams --------------------------------------------
P = Ex01Parameters(profile="tiny")                 # or "wide" / "tiny" / "debug"
P.assert_consistent()

trainer = P.trainer_defaults()  

# --- Data --------------------------------------------------------------------
train_valid_data = rom.FetchTrainAndValidSet(0.8, 'ex01', 'N_A_reduced')

# --- DOD model ---------------------------------------------------------------
DOD_model = rom.innerDOD(**P.make_innerDOD_kwargs())

# --- Trainer -----------------------------------------------------------------
DOD_trainer = rom.innerDODTrainer(
    P.Nt,
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
