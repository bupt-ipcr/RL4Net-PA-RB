import argparse
import utils
import sys
import numpy as np
import itertools
from datetime import datetime, timedelta
from tqdm import tqdm


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--seed', type=int, default=799345)
    parser.add_argument('-e', '--es', action='store_true')
    parser.add_argument('-c', '--card_no', type=int,
                        help='GPU card no.', default=0)
    args = parser.parse_args()
    return args


def mock_FP_algorithm(env):
    P = np.random.rand(env.n_channel)
    fading = env.fading.T
    g_ii = fading.diagonal()
    for _ in range(100):
        P_last = P
        recv_power = P * fading
        signal_power = recv_power.diagonal()
        total_power = recv_power.sum(axis=1)
        inter_noise = total_power - signal_power + \
            env.noise_mW / 1000

        gamma = signal_power / inter_noise
        y = np.sqrt((1.+gamma) * signal_power) / inter_noise
        y_j = np.tile(np.expand_dims(y, axis=1), [1, env.n_channel])
        P = np.minimum(env.max_mW / 1000, np.square(y) *
                    (1.+gamma) * g_ii / np.sum(np.square(y_j)*fading, axis=1))
        if np.linalg.norm(P_last - P) < 1e-3:
            break
    # convert with rb allocation
    action = np.zeros((env.n_channel, env.n_rb))
    for i, p in enumerate(P):
        action[i][i%env.n_valid_rb] = p * 1000  
    return action


def mock_WMMSE_algorithm(env):
    v = np.random.rand(env.n_channel)
    fading = env.fading.T
    recv_power = np.square(v) * fading
    signal_power = recv_power.diagonal()
    total_power = recv_power.sum(axis=1)
    inter_noise = total_power - signal_power + \
        env.noise_mW / 1000

    u = np.sqrt(signal_power) / total_power
    w = 1 + signal_power / inter_noise
    C = np.sum(w)
    for _ in range(100):
        C_last = C
        W = w * np.ones([env.n_channel, env.n_channel])
        U = u * np.ones([env.n_channel, env.n_channel])
        v_ = W*U**2*fading
        v = w*u*np.sqrt(fading.diagonal()) / np.sum(v_, axis=1)
        v = np.minimum(
            np.sqrt(env.max_mW / 1000), np.maximum(1e-10*np.random.rand(env.n_channel), v))

        recv_power = np.square(v) * fading
        signal_power = recv_power.diagonal()
        total_power = recv_power.sum(axis=1)
        inter_noise = total_power - signal_power + \
            env.noise_mW / 1000

        u = np.sqrt(signal_power) / total_power
        w = 1 + signal_power / inter_noise
        C = np.sum(w)
        if np.abs(C_last - C) < 1e-3:
            break
    P = v**2
    # convert with rb allocation
    action = np.zeros((env.n_channel, env.n_rb))
    for i, p in enumerate(P):
        action[i][i%env.n_valid_rb] = p * 1000
    return action


def random_algorithm(env):
    n_actions = env.n_actions
    return np.random.randint(0, n_actions, env.n_pair)


def maximum_algorithm(env):
    # 轮询使用信道且最大
    n_valid_rb = env.n_valid_rb
    # action dims: n_t * n_rb
    action = np.zeros(env.n_pair, dtype=np.int32)
    max_power_level = int(env.n_actions / env.n_valid_rb)
    for i in range(env.n_pair):
        cur_rb = i % n_valid_rb
        a = max_power_level * (cur_rb + 1) - 1
        action[i] = a
    return action


def fullmax_algorithm(env):
    # 全发最大
    action = np.full((env.n_channel, env.n_rb), env.max_mW)
    return action

def fullrandom_algorithm(env):
    # 全发随机
    action = np.random.uniform(env.min_mW, env.max_mW, size=(env.n_channel, env.n_rb))
    return action


def exhausted_algorithm(env):
    # 每轮最多允许计算2s
    step = env.cur_step

    env.reset()
    cur = datetime.now()
    best_rate, best_action = 0, None
    # for action in tqdm((np.array(a, dtype=np.int32) for a  in itertools.product(range(env.n_actions), repeat=env.n_pair))):
    for action in (np.array(a, dtype=np.int32) for a  in itertools.product(range(env.n_actions), repeat=env.n_pair)):
        env.cur_step = step
        s, r, d, i = env.step(action, unit="dBm")
        rate = i['rate']
        if rate > best_rate:
            best_rate, best_action = rate, action
        # if datetime.now() - cur > timedelta(seconds=10):
            # print('exhausted time > 2s, exit')
            # sys.exit(1)
    env.cur_step = step
    return best_action


def cal_benchmark(algorithm, env):
    env.reset()
    cum_rate = []
    if algorithm.name == 'exhausted':
        t = tqdm(desc=f"Calculating {algorithm.name}:", total=env.Ns)
    while True:
        if algorithm.name == 'exhausted': t.update()
        p = algorithm.func(env)
        s_, r, d, i = env.step(p, unit=algorithm.unit)
        rate = i['rate']
        cum_rate.append(rate)
        if d:
            return algorithm.name, np.mean(cum_rate)


def cal_benchmarks(env, args=None):
    from collections import namedtuple
    Algorithm = namedtuple('Algorithm', 'name func unit')
    algorithms = [
        Algorithm('fp', mock_FP_algorithm, 'mW'),
        Algorithm('wmmse', mock_WMMSE_algorithm, 'mW'),
        Algorithm('random', random_algorithm, 'dBm'),
        Algorithm('maximum', maximum_algorithm, 'dBm'),
        Algorithm('fullrandom', fullrandom_algorithm, 'mW'),
        Algorithm('fullmax', fullmax_algorithm, 'mW'),
    ]
    if args and args.es:
        algorithms.append(Algorithm('exhausted', exhausted_algorithm, 'dBm'))
    results = []
    for algorithm in algorithms:
        yield cal_benchmark(algorithm, env)


if __name__ == '__main__':
    args = get_args()
    env = utils.get_env(seed=args.seed)
    results = cal_benchmarks(env, args=get_args())
    for name, reward in results:
        print(f'{name}: {reward}', flush=True)
