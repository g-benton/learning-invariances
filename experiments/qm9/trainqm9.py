import torch
import lie_conv.lieGroups as lieGroups
import functools
import copy
import pandas as pd
from augerino.models.qm9_models import makeTrainer
import copy
from oil.tuning.study import train_trial,Study
from oil.tuning.args import argupdated_config
import lie_conv.moleculeTrainer as moleculeTrainer
# Example run single with argument parsing
if __name__=='__main__':
    Trial = train_trial(makeTrainer)
    defaults = copy.deepcopy(makeTrainer.__kwdefaults__)
    defaults['trainer_config']['early_stop_metric']='valid_MAE'
    defaults['save']=False
    cfg,outcome = Trial(argupdated_config(defaults,namespace=(moleculeTrainer,lieGroups)))
    print(outcome)