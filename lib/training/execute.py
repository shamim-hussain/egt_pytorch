import os
import yaml
from yaml import SafeLoader as yaml_Loader
import numpy as np
import torch
import random
from lib.training.training import read_config_from_file
import importlib

MASTER_ADDR = 'localhost'
MASTER_PORT = '12356'

SCHEME_LIB = 'lib.training.schemes'
SCHEME_CLS = 'SCHEME'

KEY_SCHEME = 'scheme'
KEY_SEED = 'random_seed'
KEY_DISTRIBUTED = 'distributed'

COMMANDS = {
    'train': 'execute_training',
    'predict': 'make_predictions',
    'evaluate': 'do_evaluations',
}

DEFAULT_CONFIG_FILE = 'config_input.yaml'

def get_configs_from_args(args):
    config = {}
    args = args[1:].copy()
    
    if os.path.isfile(args[0]):
        config.update(read_config_from_file(args[0]))
        args = args[1:]
    elif os.path.isdir(args[0]):
        config_path = os.path.join(args[0], 'config_input.yaml')
        config.update(read_config_from_file(config_path))
        args = args[1:]
    
    if len(args)>0:
        additional_configs = yaml.load('\n'.join(args), 
                                       Loader=yaml_Loader)
        config.update(additional_configs)
    
    if not KEY_SCHEME in config:
        raise ValueError(f'"{KEY_SCHEME}" is not in config!')
    return config

def import_scheme(scheme_name):
    full_name = f'{SCHEME_LIB}.{scheme_name}.{SCHEME_CLS}'
    module_name, object_name = full_name.rsplit('.', 1)
    imported_module = importlib.import_module(module_name)
    return getattr(imported_module, object_name)


def run_worker(rank, world_size, command, scheme_class, config, seed):
    torch.cuda.set_device(rank)
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.distributed.init_process_group(backend="nccl", 
                                         rank=rank,
                                         world_size=world_size)
    
    print(f'Initiated rank: {rank}', flush=True)
    try:
        scheme = scheme_class(config, rank, world_size)
        getattr(scheme, COMMANDS[command])()
    finally:
        torch.distributed.destroy_process_group()
        print(f'Rank {rank}:Destroyed process!', flush=True)
    

def execute(command, config):
    scheme_class = import_scheme(config[KEY_SCHEME])
    
    world_size = torch.cuda.device_count()
    
    if KEY_SEED in config and config[KEY_SEED] is not None:
        seed = config[KEY_SEED]
    else:
        seed = random.randint(0, 100000)
    
    if KEY_DISTRIBUTED in config and config[KEY_DISTRIBUTED] and world_size>1:
        os.environ['MASTER_ADDR'] = MASTER_ADDR
        os.environ['MASTER_PORT'] = MASTER_PORT
        torch.multiprocessing.spawn(fn = run_worker,
                                    args = (world_size,command,scheme_class,config,seed),
                                    nprocs = world_size,
                                    join = True)
    else:
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        scheme = scheme_class(config)
        getattr(scheme, COMMANDS[command])()
        
