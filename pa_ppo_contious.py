#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@author: Jiawei Wu
@create time: 1970-01-01 08:00
@edit time: 2021-04-01 17:09
@file: /RL4Net-PA-RB/pa_ppo_contious.py
@desc: 
"""
import utils
import numpy as np
from policy_el_ppo import DiscrateAgentAdapter as PPO
from torch.utils.tensorboard import SummaryWriter
from benckmarks import cal_benchmarks
from argparse import ArgumentParser
import json
MAX_EPISODES = 1000
DECAY_THRES = 500

POLICY_DESC = "PPO"
TRAIN_EVERY = 1024


@utils.timeit
def dqn_loop(env, agent, logdir):
    summary_writer = SummaryWriter(log_dir=logdir)
    # train
    print(f"Start {POLICY_DESC} loop.")
    train_his = []
    run_steps = 0
    for ep in range(MAX_EPISODES):
        cur_state = env.reset()
        cur_state = cur_state.reshape((-1, env.n_states))
        done = False
        ep_his = []
        agent.epsilon = max((DECAY_THRES - ep) / DECAY_THRES, 0.001)
        while True:
            run_steps += 1
            action, noise = agent.get_action(cur_state)
            # process action
            action = np.clip(action, 0, env.max_mW)
            next_state, reward, done, info = env.step(
                action, unit='mW')
                
            next_state = next_state.reshape((-1, env.n_states))
            agent.add_steps(cur_state, action, reward, done, noise)

            if run_steps % TRAIN_EVERY == 0:
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
                    print('EP: ', len(train_his),  f'{POLICY_DESC}:',
                          np.mean([t['cum_reward'] for t in train_his[-10:]]), info, flush=True)
                break

    # find best ep_his
    train_his.sort(key=lambda o: o['cum_reward'], reverse=True)
    dqn_result = train_his[0]['cum_reward'], train_his[0]['ep_his']
    return dqn_result


def get_dqn_agent(env, **kwargs):
    n_states = env.n_states
    n_actions = env.n_valid_rb
    agent = PPO(n_states, n_actions, if_discrete=False, **kwargs)
    return agent


def get_instances(args=utils.get_args()):
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


if __name__ == '__main__':
    instances = get_instances()
    demo(*instances)
