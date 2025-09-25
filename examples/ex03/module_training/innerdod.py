import torch
from core import reduced_order_models as rom
from core.configs.parameters import Ex03Parameters
from utils.paths import state_dicts_path

# --- Configure run / hyperparams --------------------------------------------
example_name = 'ex03'
P = Ex03Parameters(profile="baseline")                
P.assert_consistent()

trainer = P.trainer_defaults()  

# --- Data --------------------------------------------------------------------
train_valid_data = rom.FetchTrainAndValidSet(0.8, example_name, 'N_A_reduced')

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
torch.save(DOD_model.state_dict(), state_dicts_path(example_name) / f'DOD_Module.pth')
