from core.bindings.dune_classes import *

from dune.darcyflow import _darcyflow as df

def discretize(config_dict):
  solver = df.DarcyFlowSolver(config_dict)
  product = {"l2": DuneL2Product(solver)}
  return DuneDarcyFlowModel(solver, config_dict, products=product)
