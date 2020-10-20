import json
import re
from argparse import ArgumentParser
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

here = Path()


def get_args():
    parser = ArgumentParser()
    parser.add_argument('-d', '--dir', type=str, default='runs-n_t/n_t_devices=5&sorter=power',
                        help='Directory to visualize.')
    args = parser.parse_args()
    if not args.dir.startswith('runs'):
        args.dir = 'runs-' + args.dir
    return args


def get_data(path: Path()):
    data = {}
    with path.open() as f:
        for line in f.readlines():
            if re.match(r'^[a-z]+: [\d.]+$', line):
                algorithm, reward = line.split(': ')
                reward = float(reward)
                data[algorithm] = reward
    return data


def get_datas(root: Path()):
    datas = []
    for path in root.iterdir():
        if path.is_dir():
            datas.extend(get_datas(path))
        elif path.name == 'results.log':
            datas.append(get_data(path))
    return datas


def plot_datas(datas):
    fp_data = np.array([o['fp'] for o in datas])
    wmmse_data = np.array([o['wmmse'] for o in datas])
    random_data = np.array([o['random'] for o in datas])
    maximum_data = np.array([o['maximum'] for o in datas])
    dqn_data = np.array([o['dqn'] for o in datas])
    all_data = [fp_data, wmmse_data, random_data, maximum_data, dqn_data]
    for data in all_data:
        plt.plot(data)
    plt.legend(['fp', 'wmmse', 'random', 'maximum', 'dqn'])
    plt.savefig(f'vis.png')
    plt.cla()

    # for data in all_data:
    #     plt.boxplot(data)
    plt.boxplot(all_data)
    plt.legend(['fp', 'wmmse', 'random', 'maximum', 'dqn'])
    plt.savefig(f'box.png')
    plt.cla()
    


if __name__ == "__main__":
    args = get_args()
    rootdir = here / args.dir
    datas = get_datas(rootdir)
    plot_datas(datas)
