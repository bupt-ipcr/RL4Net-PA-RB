import json
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml
from vvlab.envs.power_allocation.pa_env import PAEnv

config_path = 'config.yaml'
default_config_path = 'default_config.yaml'


def create_seeds():
    init_seed = (19980615 * 19970711) % 1059999
    seed_count = 100
    np.random.seed(init_seed)
    seeds = list(np.random.randint(9999, 1059999, seed_count))
    save_path = Path('seed.json')
    with save_path.open('w') as f:
        json.dump(str(seeds), f)
    return seeds


def get_config(path):
    with open(path, 'r') as f:
        config = yaml.safe_load(f)
        return config


def get_env(**kwargs):
    """默认使用config.yaml，但是kwargs优先级更高"""
    config = get_config(config_path)
    env_config = config['env']
    env_config.update(kwargs)
    env = PAEnv(**env_config)
    return env


def get_logdir(c1=None, c2=None) -> Path():
    config = c1 if c1 else get_config(config_path)
    default_config = c2 if c2 else get_config(default_config_path)
    env_diffs = {
        k: v for k, v in config['env'].items()
        if config['env'].get(k) != default_config['env'].get(k, None)
        and k != 'seed'
    }
    agent_diffs = {
        k: v for k, v in config['agent'].items()
        if config['agent'].get(k) != default_config['agent'].get(k, None)
    }
    env_diffs.update(agent_diffs)
    diffs = env_diffs

    rootdir = ''
    for k, v in diffs.items():
        if isinstance(v, list):
            v = '+'.join(str(vi) for vi in v)
        rootdir += f'{k}={v}&'
    rootdir = rootdir[:-1] if rootdir else 'default'

    rootdir = Path('runs') / rootdir
    if not rootdir.exists():
        rootdir.mkdir(parents=True)

    seed = config['env']['seed']
    seeddir = f'seed={seed}'

    now = datetime.now()
    nowdir = '_'.join(now.ctime().split(' ')[1:-1])

    logdir = rootdir / seeddir / nowdir
    return logdir
