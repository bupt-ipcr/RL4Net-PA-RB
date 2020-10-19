from argparse import ArgumentParser
from pa_dqn import DQN
from pa_main import rl_loop
import utils
from torch.utils.tensorboard import SummaryWriter
from copy import deepcopy


def get_args():
    parser = ArgumentParser()
    parser.add_argument('-d', '--default_changes', type=str,
                        action='append',
                        default=['sorter=power'],
                        help='Default changes of default config.')
    parser.add_argument('-k', '--key', type=str,
                        help='Current change key',
                        default='metrics')
    parser.add_argument('-v', '--values', type=float, nargs='+',
                        help='Values of key for iteration',
                        default=[
                            ['power', 'rate', 'fading'],
                            ['power'], ['rate'], ['fading'],
                            ['power', 'rate'], ['power', 'fading'],
                            ['rate', 'fading'],
                        ])
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
    # process int/float
    values = args.values
    if args.key != 'metrics':
        if all(int(value) == int(value) for value in values):
            args.values = [int(value) for value in values]

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
