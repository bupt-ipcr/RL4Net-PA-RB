from inspect import isfunction
import json
import re

from matplotlib.pyplot import plot
import utils
from argparse import ArgumentParser
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from tqdm import tqdm, trange
import pickle

here = Path()
figs = here / 'figs'
valid_keys = ['m_r_devices', 'n_t_devices', 'm_usrs', 'bs_power', 'batch_size']

plot_funcs = {}


def register(func):
    plot_funcs[func.__name__[5:]] = func
    return func


def get_default_config():
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
    return config


def check_and_savefig(path: Path()):
    if not path.parent.exists():
        path.parent.mkdir(parents=True)
    plt.savefig(path)


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
                        sub_datas.append({
                            'algorithm': algorithm,
                            'rate': rate
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
        n_recvs = conf['n_t_devices'] * conf['m_r_devices'] +\
            conf['n_bs'] * conf['m_usrs']
        if logdir.name != 'default':
            changes = logdir.name.split('&')
            for change in changes:
                key, value = change.split('=')
                if key != 'card_no':
                    try:
                        conf[key] = float(value)
                    except:
                        conf[key] = value
        datas = get_datas(logdir)
        for data in datas:
            data.update(conf)
            data['sum_rate'] = data['rate'] * n_recvs
        if not 2 < conf['bs_power'] < 40:
            continue
        all_data.extend(datas)
    # save
    all_data = pd.DataFrame(all_data)
    all_data.to_pickle(str(save_file))
    return all_data


@register
def plot_box(all_data):
    from functools import reduce
    from operator import and_
    dft_config = get_default_config()
    for key in tqdm(valid_keys, desc="Ploting Box"):
        for aim in ['rate', 'sum_rate']:
            fig = plt.figure(figsize=(15, 10))
            cur_index = reduce(and_, (all_data[k] == v for k, v in dft_config.items(
            ) if k in valid_keys and k != key))
            sns.boxplot(x=key, y=aim, hue="algorithm", hue_order=['dqn', 'fp', 'wmmse', 'maximum', 'random'],
                        data=all_data[cur_index], palette="Set3", showfliers=False)
            if key == 'bs_power':
                plt.xlabel(f'{key}/W')
            plt.ylabel(f'Average {aim}(bps/Hz)')
            check_and_savefig(figs / f'box/{aim}-{key}.png')
            plt.close(fig)


@register
def plot_cdf(all_data):
    from functools import reduce
    from operator import and_
    dft_config = get_default_config()
    for key in tqdm(valid_keys, desc="Ploting CDF"):
        for aim in ['rate', 'sum_rate']:
            fig = plt.figure(figsize=(15, 10))
            cur_index = reduce(and_, (all_data[k] == v for k, v in dft_config.items(
            ) if k in valid_keys and k != key))
            sns.displot(data=all_data[cur_index], x=aim, kind="ecdf", hue="algorithm", hue_order=[
                        'dqn', 'fp', 'wmmse', 'maximum', 'random'])
            if key == 'bs_power':
                plt.xlabel(f'{key}/W')
            plt.ylabel(f'Average {aim}(bps/Hz)')
            check_and_savefig(figs / f'cdf/{aim}-{key}.png')
            plt.close(fig)


@register
def plot_avg(all_data):
    from functools import reduce
    from operator import and_
    dft_config = get_default_config()
    for key in tqdm(valid_keys, desc="Ploting AVG"):
        for aim in ['rate', 'sum_rate']:
            fig = plt.figure(figsize=(15, 10))
            cur_index = reduce(and_, (all_data[k] == v for k, v in dft_config.items(
            ) if k in valid_keys and k != key))
            ax = sns.lineplot(data=all_data[cur_index], x=key, y=aim, hue="algorithm",
                              hue_order=['dqn', 'fp', 'wmmse',
                                         'maximum', 'random'],
                              style="algorithm", markers=True, dashes=False, ci=None)
            plt.xticks(sorted(list(set(all_data[cur_index][key]))))
            if key == 'bs_power':
                plt.xlabel(f'{key}/W')
            plt.ylabel(f'Average {aim}(bps/Hz)')
            check_and_savefig(figs / f'avg/{aim}-{key}.png')
            plt.close(fig)


@register
def plot_env(*args):
    seed = 312691
    env = utils.get_env(seed=seed, n_t_devices=3,
                        m_r_devices=4, m_usrs=4, R_dev=0.25)
    import matplotlib.patches as mpatches

    def cir_edge(center, radius, color):
        patch = mpatches.Circle(center, radius, fc='white', ec=color, ls='--')
        return patch

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)

    # draw d2d pairs
    for t_idx, pair in env.devices.items():
        t, rs = pair['t_device'], pair['r_devices']
        # draw edge
        ax.add_patch(cir_edge((t.x, t.y), env.R_dev, 'green'))
        # draw t device
        ax.scatter([t.x], [t.y], marker='s', s=100, c='green', zorder=10)
        # draw r devices
        for r_idx, r in rs.items():
            ax.scatter([r.x], [r.y], marker='o', s=60, c='green', zorder=10)

    # draw cell and bs
    cell_xs = env.R_bs * \
        np.array([0, np.sqrt(3)/2, np.sqrt(3)/2,
                0, -np.sqrt(3)/2, -np.sqrt(3)/2, 0])
    cell_ys = env.R_bs * np.array([1, .5, -.5, -1, -.5, .5, 1])
    ax.plot(cell_xs, cell_ys, color='black')

    ax.scatter([0.0], [0.0], marker='^', s=100, c='blue', zorder=30)
    # draw usrs
    for idx, usr in env.users.items():
        ax.scatter([usr.x], [usr.y], marker='x', s=100, c='orange', zorder=20)
        ax.plot([0, usr.x], [0, usr.y], ls='--', c='blue', zorder=20)

    check_and_savefig(figs / f'env/{seed}.png')
    plt.close(fig)


@register
def plot_all(all_data):
    for name, func in plot_funcs.items():
        if name != 'all':
            func(all_data)


if __name__ == "__main__":
    args = get_args()
    all_data = get_all_data(args)
    for attr in dir(args):
        if not attr.startswith('_') and args.__getattribute__(attr):
            func = plot_funcs.get(attr, None)
            if func:
                func(all_data)
