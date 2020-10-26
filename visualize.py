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
    parser = ArgumentParser()
    parser.add_argument('-d', '--dir', type=str, default='runs',
                        help='Directory to visualize.')
    args = parser.parse_args(args=[])
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


def get_all_data():
    args = get_args()
    runsdir = here / args.dir
    config = get_default_config()
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


def plot_box(all_data):
    from functools import reduce
    from operator import and_
    dft_config = get_default_config()
    for key in tqdm(['m_r_devices', 'n_t_devices', 'm_usrs', 'bs_power', 'batch_size'], desc="Ploting"):
        for aim in ['rate', 'sum_rate']:
            fig = plt.figure(figsize=(15, 10))
            cur_index = reduce(and_, (all_data[k] == v for k, v in dft_config.items() if k in [
                'm_r_devices', 'n_t_devices', 'm_usrs', 'bs_power', 'batch_size'] and k != key))
            sns.boxplot(x=key, y=aim, hue="algorithm", hue_order=['dqn', 'fp', 'wmmse', 'maximum', 'random'],
                        data=all_data[cur_index], palette="Set3", showfliers=False)
            check_and_savefig(figs / f'box/{aim}-{key}.png')
            plt.close()


def plot_all():
    all_data = get_all_data()
    plot_box(all_data)


if __name__ == "__main__":
    plot_all()
