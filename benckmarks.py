import argparse
import utils
import numpy as np


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--seed', type=int, default=799345)
    args = parser.parse_args()
    return args


def mock_FP_algorithm(env):
    P = np.random.rand(env.n_recvs)
    loss = env.loss
    g_ii = loss.diagonal()
    for _ in range(100):
        P_last = P
        recv_power = P * loss
        signal_power = recv_power.diagonal()
        total_power = recv_power.sum(axis=1)
        inter_noise = total_power - signal_power + \
            1e-3*pow(10., env.thres_power/10.)

        gamma = signal_power / inter_noise
        y = np.sqrt((1.+gamma) * signal_power) / inter_noise
        y_j = np.tile(np.expand_dims(y, axis=1), [1, env.n_recvs])
        P = np.minimum(env.power_levels[-1], np.square(y) *
                       (1.+gamma) * g_ii / np.sum(np.square(y_j)*loss, axis=1))
        if np.linalg.norm(P_last - P) < 1e-3:
            break
    return P


def mock_WMMSE_algorithm(env):
    v = np.random.rand(env.n_recvs)  # max_power*np.ones((N))
    loss = env.loss
    recv_power = np.square(v) * loss
    signal_power = recv_power.diagonal()
    total_power = recv_power.sum(axis=1)
    inter_noise = total_power - signal_power + \
        1e-3*pow(10., env.thres_power/10.)

    u = np.sqrt(signal_power) / total_power
    w = 1 + signal_power / inter_noise
    C = np.sum(w)
    for _ in range(100):
        C_last = C
        W = w * np.ones([env.n_recvs, env.n_recvs])
        U = u * np.ones([env.n_recvs, env.n_recvs])
        v_ = W*U**2*loss
        v = w*u*np.sqrt(loss.diagonal()) / np.sum(v_, axis=1)
        v = np.minimum(
            np.sqrt(env.power_levels[-1]), np.maximum(1e-10*np.random.rand(env.n_recvs), v))

        recv_power = np.square(v) * loss
        signal_power = recv_power.diagonal()
        total_power = recv_power.sum(axis=1)
        inter_noise = total_power - signal_power + \
            1e-3*pow(10., env.thres_power/10.)

        u = np.sqrt(signal_power) / total_power
        w = 1 + signal_power / inter_noise
        C = np.sum(w)
        if np.abs(C_last - C) < 1e-3:
            break
    P = v**2
    return P


def random_algorithm(env):
    n_actions = env.n_actions
    return np.random.randint(0, n_actions, env.n_t * env.m_r)


def maximum_algorithm(env):
    n_actions = env.n_actions
    return np.full(env.n_t * env.m_r, n_actions - 1)


def cal_benchmark(algorithm, env):
    env.reset()
    cum_r = 0
    while True:
        p = algorithm.func(env)
        raw = algorithm.type == 'power'
        s_, r, d, i = env.step(p, raw=raw)
        cum_r += r/env.n_recvs
        if d:
            return algorithm.name, cum_r/env.n_recvs


def cal_benchmarks(env):
    from collections import namedtuple
    Algorithm = namedtuple('Algorithm', 'name func type')
    algorithms = [
        Algorithm('fp', mock_FP_algorithm, 'power'),
        Algorithm('wmmse', mock_WMMSE_algorithm, 'power'),
        Algorithm('random', random_algorithm, 'action'),
        Algorithm('maximum', maximum_algorithm, 'action')
    ]
    results = []
    for algorithm in algorithms:
        results.append(cal_benchmark(algorithm, env))
    return results


if __name__ == '__main__':
    args = get_args()
    env = utils.get_env(seed=args.seed)
    results = cal_benchmarks(env)
    for name, reward in results:
        print(f'{name}: {reward}')
