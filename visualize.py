import json
import re
import utils
from argparse import ArgumentParser
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from tqdm import tqdm


here = Path()
figs = here / 'figs'


def check_and_savefig(path: Path()):
    if not path.parent.exists():
        path.parent.mkdir(parents=True)
    plt.savefig(path)


def get_args():
    parser = ArgumentParser()
    parser.add_argument('-d', '--dir', type=str, default='runs',
                        help='Directory to visualize.')
    args = parser.parse_args()
    return args


def get_data(path: Path()):
    data = []
    with path.open() as f:
        for line in f.readlines():
            if re.match(r'^[a-z]+: [\d.]+$', line):
                algorithm, rate = line.split(': ')
                rate = float(rate)
                data.append({
                    'algorithm': algorithm,
                    'rate': rate
                })
    return data


def get_datas(directory: Path()):
    datas = []
    for path in directory.iterdir():
        if path.is_dir():
            datas.extend(get_datas(path))
        elif path.name == 'results.log':
            datas.extend(get_data(path))
    return datas


def get_all_data():
    args = get_args()
    runsdir = here / args.dir
    dft_config = utils.get_config('default_config.yaml')
    config = dft_config['env']
    for k, v in config.items():
        if isinstance(v, list):
            config[k] = '+'.join(v)
        try:
            config[k] = int(v)
        except:
            pass
    all_data = []
    for logdir in tqdm(list(runsdir.iterdir()), desc="Gathering all data"):
        conf = config.copy()
        if logdir.name == 'default':
            pass
        else:
            changes = logdir.name.split('&')
            for change in changes:
                key, value = change.split('=')
                if key == 'card_no':
                    continue
                try:
                    conf[key] = int(value)
                except:
                    conf[key] = value
        datas = get_datas(logdir)
        for data in datas:
            data.update(conf)
            data['sum_rate'] = data['rate'] * \
                (conf['n_t_devices'] * conf['m_r_devices'] +
                 conf['n_bs'] * conf['m_usrs'])
        all_data.extend(datas)
    return pd.DataFrame(all_data)


def plot_datas(datas, suffix=''):
    dqn_data = np.array([o['dqn'] for o in datas])
    fp_data = np.array([o['fp'] for o in datas])
    wmmse_data = np.array([o['wmmse'] for o in datas])
    random_data = np.array([o['random'] for o in datas])
    maximum_data = np.array([o['maximum'] for o in datas])
    all_data = [dqn_data, fp_data, wmmse_data, random_data, maximum_data]
    for data in all_data:
        try:
            plt.plot(data)
        except:
            print(data)
    plt.legend(['dqn', 'fp', 'wmmse', 'random', 'maximum'])
    plt.savefig(f'figs/vis{suffix}.png')
    plt.cla()

    plt.boxplot(all_data)
    plt.legend(['dqn', 'fp', 'wmmse', 'random', 'maximum'])
    plt.savefig(f'figs/box{suffix}.png')
    plt.cla()


def plot_box(all_data):
    for key in tqdm(['m_r_devices', 'n_t_devices', 'm_usrs', 'bs_power'], desc="Ploting"):
        for aim in ['rate', 'sum_rate']:
            plt.figure(figsize=(15, 10))
            ax = sns.boxplot(x=key, y=aim, hue="algorithm", hue_order=['dqn', 'fp', 'wmmse', 'maximum', 'random'],
                             data=all_data, palette="Set3", showfliers=False)
            check_and_savefig(figs / f'box/{aim}-{key}.png')
            plt.cla()


def plot_all():
    all_data = get_all_data()
    plot_box(all_data)


if __name__ == "__main__":
    plot_all()
