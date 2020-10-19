from argparse import ArgumentParser
from pa_dqn import DQN
from pa_main import rl_loop
import utils
from torch.utils.tensorboard import SummaryWriter
from copy import deepcopy
import json

keywords = [
    'seed', 'n_levels',
    'n_t_devices', 'm_r_devices', 'm_usrs', 'bs_power',
    'R_bs', 'R_dev', 'r_bs', 'r_dev',
    'sorter', 'metrics',
    'gamma', 'learning_rate', 'init_epsilon', 'min_epsilon'
]
ranges = {
    'n_t_devices': [3, 5, 7, 9, 11],
    'm_r_devices': [1, 2, 3, 4, 5],
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
                        action='append', choices=keywords,
                        default=['sorter=power'],
                        help='Default changes of default config.')
    parser.add_argument('-k', '--key', type=str, choices=keywords,
                        help='Parameter(key) which changes',
                        default='metrics')
    parser.add_argument('-v', '--values', type=str,
                        help='Values for key to range, use default if not set.'\
                            'Example: -v "[1, 2, 3, 4 , 5]"',
                        default='')
    args = parser.parse_args()

    # process default changes
    dft = {}
    for change in args.default_changes:
        key, value = change.split('=')
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
    for k in diffs.keys():
        print(k, ':', env.__getattribute__(k))

    n_states = env.n_states
    n_actions = env.n_actions
    agent = DQN(n_states, n_actions)
    # different to pa_main
    logdir = utils.get_logdir(conf, default_conf)
    return env, agent, logdir


if __name__ == '__main__':
    args = get_args()
    key, values = args.key, args.values
    dft = args.dft
    for value in values:
        cur = dft.copy()
        cur.update({key: value})
        from datetime import datetime
        start = datetime.now()
        instances = get_instance(cur)
        rl_loop(*instances)
        end = datetime.now()
        print('cost time:', end - start)
