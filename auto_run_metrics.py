import json
from argparse import ArgumentParser
from copy import deepcopy

from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import utils
from pa_dqn import DQN
from pa_main import rl_loop

keywords = [
    'seed', 'n_levels',
    'n_t_devices', 'm_r_devices', 'm_usrs', 'bs_power',
    'R_bs', 'R_dev', 'r_bs', 'r_dev',
    'sorter', 'metrics',
    'gamma', 'learning_rate', 'init_epsilon', 'min_epsilon',
    'card_no'
]
ranges = {
    'n_t_devices': [9, 11, 13, 15, 17],
    'm_r_devices': [2, 3, 4, 5, 6],
    'm_usrs': [2, 3, 4, 5, 6],
    'bs_power': [4, 6, 8, 10, 12],
    'sorter': ['power', 'rate', 'fading'],
    'metrics': [
        ['power', 'rate', 'fading'],
        ['power', 'fading'], ['fading'],
    ]
}


def get_args():
    parser = ArgumentParser()
    parser.add_argument('-d', '--default_changes', type=str,
                        action='append',
                        default=['sorter=power'],
                        help='Default changes of default config.')
    parser.add_argument('-k', '--key', type=str, choices=keywords,
                        help='Parameter(key) which changes',
                        default='n_t_devices')
    parser.add_argument('-v', '--values', type=str,
                        help='Values for key to range, use default if not set.'
                        'Example: -v "[1, 2, 3, 4, 5]"',
                        default='')
    parser.add_argument('-s', '--seeds', type=int,
                        help='Seed count.', default=100)
    parser.add_argument('-c', '--card_no', type=int,
                        help='GPU card no.', default=0)
    args = parser.parse_args()

    # process default changes
    dft = {}
    for change in args.default_changes:
        key, value = change.split('=')
        if key not in keywords:
            raise ValueError(f'Default change key should in {keywords}, but {key}')
        try:
            value = int(value)
        except:
            pass
        dft.update({key: value})
    args.dft = dft
    # process values
    if args.values:
        args.values = json.loads(args.values)
    else:
        args.values = ranges[args.key]

    return args


def recursive_merge(origin, diffs):
    for key, value in diffs.items():
        for k, v in origin.items():
            if key == k:
                origin[k] = value
                return
        for k, v in origin.items():
            if isinstance(v, dict):
                recursive_merge(v, {key: value})


def get_instance(diffs):
    default_conf = utils.get_config('default_config.yaml')
    conf = deepcopy(default_conf)
    recursive_merge(conf, diffs)

    env = utils.get_env(**conf['env'])
    # for k in diffs.keys():
    #     print(k, ':', env.__getattribute__(k))

    n_states = env.n_states
    n_actions = env.n_actions
    agent = DQN(n_states, n_actions, **conf['agent'])
    # different to pa_main
    logdir = utils.get_logdir(conf, default_conf)
    return env, agent, logdir


if __name__ == '__main__':
    args = get_args()
    key, values = args.key, args.values
    dft = args.dft
    seeds = utils.create_seeds()
    # iter values of key
    for value in values:
        cur = dft.copy()
        # iter seeds
        for seed in seeds[:args.seeds]:
            cur.update({key: value})
            cur.update({'seed': seed})
            instances = get_instance(cur)
            rl_loop(*instances)
