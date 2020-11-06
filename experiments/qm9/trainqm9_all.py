import torch
import lie_conv.lieGroups as lieGroups
import functools
import copy
import pandas as pd
from augerino.models.qm9_models import makeTrainer
import copy
from oil.tuning.study import train_trial,Study


expt_settings = {'auger':{'augerino':True,'aug':True},
                'base':{'augerino':False,'aug':False},
                'aug':{'augerino':False,'aug':True}}

if __name__ == '__main__':
    Trial = train_trial(makeTrainer)
    config_spec = copy.deepcopy(makeTrainer.__kwdefaults__)
    config_spec['trainer_config']['log_suffix']='qm9_augerino'
    config_spec['trainer_config']['early_stop_metric']='valid_MAE'
    config_spec.update({'task':['homo','lumo']})
    config_spec['net_config']['group']=[lieGroups.Trivial(3),lieGroups.T(3)]
    name = 'qm9_augerino_expt_full'#config_spec.pop('study_name')
    thestudy = Study(Trial,{},study_name=name,base_log_dir=config_spec['trainer_config'].get('log_dir',None))
    for name,net_cfg in expt_settings.items():
        the_config = copy.deepcopy(config_spec)
        the_config['net_config'].update(net_cfg)
        the_config['name']=name
        thestudy.run(num_trials=-1,new_config_spec=the_config,ordered=True)
    print(thestudy.results_df())