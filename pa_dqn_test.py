#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@author: Jiawei Wu
@create time: 1970-01-01 08:00
@edit time: 2021-04-14 15:28
@file: /RL4Net-PA-RB/pa_dqn_test.py
@desc: 
"""
import utils
from tqdm import tqdm
import numpy as np
from policy_dqn import DQN
# from policy_el_dqn import DQNAgentAdapter as DQN
from torch.utils.tensorboard import SummaryWriter
from benckmarks import cal_benchmarks
from argparse import ArgumentParser
import json
MAX_EPISODES = 700
DECAY_THRES = 500
from collections import defaultdict

@utils.timeit
def dqn_loop(env, agent, logdir):
    summary_writer = SummaryWriter(log_dir=logdir)
    # train
    print(f"Start DQN loop.")
    train_his = []
    for ep in range(MAX_EPISODES):
        cur_state = env.reset()
        cur_state = cur_state.reshape((-1, env.n_states))
        done = False
        ep_his = []
        agent.epsilon = max((DECAY_THRES - ep) / DECAY_THRES, 0.001)
        while True:
            action = agent.get_action(cur_state)[0]
            next_state, reward, done, info = env.step(
                action.astype(np.int32), unit='dBm')
            next_state = next_state.reshape((-1, env.n_states))
            agent.add_steps(cur_state, action, reward, done, next_state)
            loss = agent.learn()
            if loss:
                summary_writer.add_scalar('loss', loss, agent.eval_step)
            cur_state = next_state
            ep_his.append(info['rate'])
            if done:
                cum_reward = np.mean(ep_his)
                summary_writer.add_scalar('reward', cum_reward, ep)
                train_his.append({'cum_reward': cum_reward, 'ep_his': ep_his})
                if len(train_his) % 10 == 0:
                    print('EP: ', len(train_his),  'DQN:',
                          np.mean([t['cum_reward'] for t in train_his[-10:]]), info, flush=True)
                break
    # find best ep_his
    train_his.sort(key=lambda o: o['cum_reward'], reverse=True)
    dqn_result = train_his[0]['cum_reward'], train_his[0]['ep_his']
    return dqn_result


def get_dqn_agent(env, **kwargs):
    n_states = env.n_states
    n_actions = env.n_actions
    agent = DQN(n_states, n_actions, **kwargs)
    return agent


def get_instances(args=utils.get_args()):
    args.env['seed'] = 12345
    env = utils.get_env(**args.env)
    agent = get_dqn_agent(env, **args.agent)
    conf = utils.get_config('config.yaml')
    conf['env'].update(args.env)
    conf['agent'].update(args.agent)
    logdir = utils.get_logdir(conf)
    return env, agent, logdir


def demo(env, agent, logdir):
    dqn_result = dqn_loop(env, agent, logdir)

    result_path = logdir / 'results.log'
    with result_path.open('w') as f:
        # RL results
        f.write('dqn: ' + str(dqn_result[0]) + '\r\n')
        # f.write(str(dqn_result[1]))
        # benckmarks
        results = cal_benchmarks(env)
        for result in results:
            f.write(result[0] + ': ' + str(result[1]) + '\r\n')
    print('done')


def test_with_seed(env, agent, seed):
    cur_state = env.reset(seed=seed)
    # print(f'now env.seed is {env.seed}')
    rates = []
    while True:
        action = agent.get_action(cur_state)[0]
        next_state, reward, done, info = env.step(
            action.astype(np.int32), unit='dBm')
        rates.append(info['rate'])
        cur_state = next_state.reshape((-1, env.n_states))
        if done: break
    # print(f'seed {seed}, score {np.mean(rates)}')
    return np.mean(rates)


if __name__ == '__main__':
    args = utils.get_args()
    env = utils.get_env(**args.env)
    agent = get_dqn_agent(env, **args.agent)
    agent.load('common_model.pth')

    rnd_cnt = 100
    rnds = np.random.randint(10000, 100000, rnd_cnt)
    scores = defaultdict(list)
    for rnd in tqdm(rnds):
        scores['dqn'].append(test_with_seed(env, agent, rnd))

        env.reset(seed=rnd)
        results = cal_benchmarks(env)
        for alg, score in results:
            scores[alg].append(score)
    
    for k, v in scores.items():
        print(f'{k}: {np.mean(v)}')
    