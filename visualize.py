#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@author: Jiawei Wu
@create time: 2021-01-06 20:21
@edit time: 2021-01-07 17:09
@file: /workspace/RL4Net-PA-RB/visualize.py
@desc: 
"""
from operator import and_
from functools import reduce
from inspect import isfunction
import json
import re
from tensorboard.backend.event_processing import event_accumulator
from matplotlib.pyplot import plot
import utils
from argparse import ArgumentParser
from pathlib import Path
import itertools
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from tqdm import tqdm, trange
import pickle
from collections import namedtuple, OrderedDict, deque

here = Path()
figs = here / 'figs'
valid_keys = ['Number of D2D pairs',
              'Number of CUE', 'BS Power (W)', 'Batch Size']
alias = {
    'bs_power': 'BS Power (W)', 'm_cue': 'Number of CUE',
    'n_pair': 'Number of D2D pairs', 'batch_size': 'Batch Size',
}
plot_funcs = {}


def register(func):
    plot_funcs[func.__name__[5:]] = func
    return func


def get_args():
    parser = ArgumentParser(description='Params for ploting.')
    parser.add_argument('-d', '--dir', type=str, default='runs',
                        help='Directory to visualize.')
    parser.add_argument('-r', '--reload', action='store_true',
                        help='Force to reload data.')
    for name, func in plot_funcs.items():
        parser.add_argument(f'--{name}', action='store_true',
                            help=f'Whether to plot {name}.')
    args = parser.parse_args()
    if not any(arg[1] for arg in args._get_kwargs() if arg[0] not in {'dir', 'reload'}):
        args.all = True
    return args


def get_default_config(rename=False):
    dft_config = utils.get_config('default_config.yaml')
    config = dft_config['env']
    agent_config = dft_config['agent']
    config.update(agent_config)
    for k, v in config.items():
        if isinstance(v, list):
            config[k] = '+'.join(v)
        try:
            config[k] = int(v)
        except:
            pass
    # add
    if rename:
        for k, v in alias.items():
            config[v] = config[k]
    return config


dft_config = get_default_config(rename=True)


def check_and_savefig(path: Path(), *args, **kwargs):
    if not path.parent.exists():
        path.parent.mkdir(parents=True)
    plt.savefig(path, *args, **kwargs, dpi=300)


def get_datas(directory: Path()):
    datas = []
    for path in directory.iterdir():
        if path.is_dir():
            sub_datas = get_datas(path)
        elif path.name == 'results.log':
            sub_datas = []
            with path.open() as f:
                for line in f.readlines():
                    if re.match(r'^[a-z]+: [\d.]+$', line):
                        algorithm, rate = line.split(': ')
                        rate = float(rate)
                        algorithm_mapping = {
                            'dqn': 'DRPA',
                            'fp': 'FP',
                            'wmmse': 'WMMSE',
                            'random': 'Rnd',
                            'maximum': 'Max',
                            'fullrandom': 'Full-Rnd',
                            'fullmax': 'Full-Max',
                            'madqn': 'MADRPA',
                        }
                        sub_datas.append({
                            'algorithm': algorithm_mapping[algorithm],
                            'Rate': rate
                        })
        else:
            sub_datas = []
        datas.extend(sub_datas)
    return datas


def get_all_data(args):
    runsdir = here / args.dir
    # try to load data from pickle
    save_file = here / 'all_data.pickle'
    if not args.reload and save_file.exists():
        with save_file.open('rb') as f:
            all_data = pickle.load(f)
            print('Load data from pickle.')
            return all_data
    config = get_default_config()
    all_data = []
    for logdir in tqdm(list(runsdir.iterdir()), desc="Gathering all data"):
        conf = config.copy()
        n_recvs = conf['n_pair'] + conf['n_bs'] * conf['m_cue']
        if logdir.is_dir() and logdir.name != 'default':
            changes = logdir.name.split('&')
            for change in changes:
                key, value = change.split('=')
                if key != 'card_no':
                    try:
                        conf[key] = int(value)
                    except:
                        try:
                            conf[key] = float(value)
                        except:
                            conf[key] = value
        datas = get_datas(logdir)
        for data in datas:
            data.update(conf)
            data['sum-rate'] = data['Rate'] * n_recvs
        all_data.extend(datas)
    # save
    all_data = pd.DataFrame(all_data)
    # replace column names
    all_data.rename(columns=alias,
                    inplace=True)
    all_data.to_pickle(str(save_file))
    return all_data


def get_events(logdir):
    datas = {}
    tmp = []
    for file in logdir.iterdir():
        if file.is_dir(): return get_events(list(file.iterdir())[0])
        else:
            if file.name.startswith('events'):
                ea=event_accumulator.EventAccumulator(str(file.resolve()))
                ea.Reload()
                data = np.array([i.value for i in ea.scalars.Items('reward')])
                tmp.append(data)
            else:
                assert file.name == 'results.log'
                with file.open('r') as f:
                    for line in f.readlines():
                        algorithm, rate = line.split(': ')
                        datas[algorithm] = float(rate[:-1])
    assert len(tmp) == 2
    madqn, dqn = (tmp[0], tmp[1]) if tmp[0].sum() > tmp[1].sum() else (tmp[1], tmp[0])
    for k in datas.keys(): datas[k] = np.full(len(dqn), datas[k])
    datas['dqn'] = dqn
    datas['madqn'] = madqn
    return datas

def get_events_data(args):
    runsdir = here / args.dir
    # try to load data from pickle
    save_file = here / 'events_data.pickle'
    if not args.reload and save_file.exists():
        with save_file.open('rb') as f:
            events_data = pickle.load(f)
            print('Load data from pickle.')
            return events_data
    # single, duplex, default, m_state_8, m_state_20, n_level_18, n_level14
    # default: m_state_16, n_level_10
    logdir = Path('runs-jsac')
    datas = {}
    for dir_ in logdir.iterdir():
        if 'n_level=14' in dir_.name: datas['n_level_14'] = get_events(dir_)
        elif 'n_level=18' in dir_.name: datas['n_level_18'] = get_events(dir_)
        elif 'm_state=8' in dir_.name: datas['m_state_8'] = get_events(dir_)
        elif 'm_state=20' in dir_.name: datas['m_state_20'] = get_events(dir_)
        elif 'm_cue=2' in dir_.name and 'rb=single' in dir_.name: datas['single'] = get_events(dir_)
        elif 'm_cue=2' in dir_.name and 'rb=single' not in dir_.name: datas['duplex'] = get_events(dir_)
        else: datas['m_state_16'] = datas['n_level_10'] = get_events(dir_)
    datas = pd.DataFrame(datas)
    datas.to_pickle(str(save_file))
    return datas
    

def smooth(d):
    S_RANGE = 3
    S_RATIO = 0.6
    def _smooth(d_):
        tmp = deque(maxlen=S_RANGE)
        # first traverse, get average
        d_avg  = []
        for i in d_:
            tmp.append(i)
            d_avg.append(np.mean(tmp))
        d_smooth = [(1-S_RATIO)*o + S_RATIO*m for o, m in zip(d_, d_avg)]
        return d_smooth

    for k, v in d.items():
        d[k] = _smooth(v)
    return d

@register
def plot_jsac_B_1(edata, adata):

    sns.set_style('white')
    # fig = plt.figure(figsize=(10, 7.5))
    fig = plt.figure()
    ax = sns.boxplot(data=all_data, x='algorithm', y='Rate', whis=100)
    plt.title("algorithms ranking (box)")
    plt.ylabel(f'Average Rate (bps/Hz)')
    ax.grid(axis="y")
    check_and_savefig(figs / f'jsac/B_1.png')
    plt.close(fig)

@register
def plot_jsac_B_2(edata, adata):
    single = edata['single']
    algorithms = OrderedDict({
        'ES': single['exhausted'],
        'MADQN': single['madqn'],
        'DQN': single['dqn'],
        'RR-FP': single['fp'],
        'RR-WMMSE': single['wmmse'],
        'RR-MAX': single['maximum'],
        'RR-RND': single['random'],
        'MAX': single['fullmax'],
        'RND': single['fullrandom'],
    })
    algorithms = smooth(algorithms)
    a_cnt, a_len = len(algorithms), len(algorithms[list(algorithms.keys())[0]])
    datas = pd.DataFrame({
        'step': np.tile(np.arange(a_len), a_cnt),
        'algorithm': np.repeat(list(algorithms.keys()), a_len),
        'Rate': itertools.chain.from_iterable(algorithms.values())
    })
    fig = plt.figure()
    plt.title("distributed vs. centerlized")
    plt.ylabel(f'Average Rate (bps/Hz)')
    ax = sns.lineplot(data=datas, x='step', y='Rate', hue='algorithm')
    ax.grid(axis="y")
    check_and_savefig(figs / f'jsac/B_2.png')
    plt.close(fig)

@register
def plot_jsac_C(edata, adata):
    single = edata['single']
    duplex = edata['duplex']
    algorithms = OrderedDict({
        'S-ES': single['exhausted'],
        'S-MADQN': single['madqn'],
        'D-ES': duplex['exhausted'],
        'D-MADQN': duplex['madqn'],
    })
    algorithms = smooth(algorithms)
    a_cnt, a_len = len(algorithms), len(algorithms[list(algorithms.keys())[0]])
    datas = pd.DataFrame({
        'step': np.tile(np.arange(a_len), a_cnt),
        'algorithm': np.repeat(list(algorithms.keys()), a_len),
        'Rate': itertools.chain.from_iterable(algorithms.values())
    })
    fig = plt.figure()
    plt.title("single vs. duplex")
    plt.ylabel(f'Average Rate (bps/Hz)')
    ax = sns.lineplot(data=datas, x='step', y='Rate', hue='algorithm')
    ax.grid(axis="y")
    check_and_savefig(figs / f'jsac/C.png')
    plt.close(fig)

@register
def plot_jsac_D(edata, adata):
    m_state_8 = edata['m_state_8']
    m_state_16 = edata['m_state_16']
    m_state_20 = edata['m_state_20']
    algorithms = OrderedDict({
        '8-MADQN': m_state_8['madqn'],
        '16-MADQN': m_state_16['madqn'],
        '20-MADQN': m_state_20['madqn'],
        '8-DQN': m_state_8['dqn'],
        '16-DQN': m_state_16['dqn'],
        '20-DQN': m_state_20['dqn'],
        'RR-RND': m_state_20['random']
    })
    algorithms = smooth(algorithms)
    a_cnt, a_len = len(algorithms), len(algorithms[list(algorithms.keys())[0]])
    datas = pd.DataFrame({
        'step': np.tile(np.arange(a_len), a_cnt),
        'algorithm': np.repeat(list(algorithms.keys()), a_len),
        'Rate': itertools.chain.from_iterable(algorithms.values())
    })
    fig = plt.figure()
    plt.title("m_state changes")
    plt.ylabel(f'Average Rate (bps/Hz)')
    ax = sns.lineplot(data=datas, x='step', y='Rate', hue='algorithm')
    ax.grid(axis="y")
    check_and_savefig(figs / f'jsac/D.png')
    plt.close(fig)

@register
def plot_jsac_E(edata, adata):
    n_level_10 = edata['n_level_10']
    n_level_14 = edata['n_level_14']
    n_level_18 = edata['n_level_18']
    algorithms = OrderedDict({
        '10-MADQN': n_level_10['madqn'],
        '14-MADQN': n_level_14['madqn'],
        '18-MADQN': n_level_18['madqn'],
        '10-DQN': n_level_10['dqn'],
        '14-DQN': n_level_14['dqn'],
        '18-DQN': n_level_18['dqn'],
        'RR-RND': n_level_18['random']
    })
    algorithms = smooth(algorithms)
    a_cnt, a_len = len(algorithms), len(algorithms[list(algorithms.keys())[0]])
    datas = pd.DataFrame({
        'step': np.tile(np.arange(a_len), a_cnt),
        'algorithm': np.repeat(list(algorithms.keys()), a_len),
        'Rate': itertools.chain.from_iterable(algorithms.values())
    })
    fig = plt.figure()
    plt.title("n_level changes")
    plt.ylabel(f'Average Rate (bps/Hz)')
    ax = sns.lineplot(data=datas, x='step', y='Rate', hue='algorithm')
    ax.grid(axis="y")
    check_and_savefig(figs / f'jsac/E.png')
    plt.close(fig)
    
@register
def plot_all(edata, adata):
    for name, func in plot_funcs.items():
        if name != 'all':
            func(edata, adata)


if __name__ == "__main__":
    args = get_args()
    all_data = get_all_data(args)
    events_data = get_events_data(args)
    for attr in dir(args):
        if not attr.startswith('_') and args.__getattribute__(attr):
            func = plot_funcs.get(attr, None)
            if func:
                func(events_data, all_data)



