#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@author: Jiawei Wu
@create time: 1970-01-01 08:00
@edit time: 2020-12-23 10:08
@FilePath: /PA/pa_multi_dqn.py
@desc: 
"""


import utils
import numpy as np
from pa_dqn import DQN
from torch.utils.tensorboard import SummaryWriter
from benckmarks import cal_benchmarks
MAX_EPISODES = 1000
DECAY_THRES = 400

class ListAgents:
    def __init__(self, policy, num, *args, **kwargs):
        self._step = 0
        self.agents = [policy(*args, **kwargs) for _ in range(num)]

    def get_action(self, states):   
        action = np.array([
            agent.get_action(np.array([state]))[0] for state, agent in zip(states, self.agents)
        ])
        return action.T

    def add_steps(self, cur_states, actions, reward, done, next_states):
        for cur_state, action, next_state, agent in zip(cur_states, actions, next_states, self.agents):
            agent.add_step(cur_state, action, reward, done, next_state)

    def learn(self):
        self._step += 1
        losses = []
        for agent in self.agents:
            loss = agent.learn()
            if loss: losses.append(loss)
        return np.mean(losses) if losses else None

    @property
    def eval_step(self):
        return self._step

    @property
    def epsilon(self):
        return 0

    @epsilon.setter
    def epsilon(self, _epsilon):
        for agent in self.agents:
            agent.epsilon = _epsilon

def rl_loop(env, agents, logdir):
    summary_writer = SummaryWriter(log_dir=logdir)
    # train
    train_his = []
    for ep in range(MAX_EPISODES):
        cur_state = env.reset()
        cur_state = cur_state.reshape((-1, env.n_states))
        done = False
        ep_his = []
        agents.epsilon = max((DECAY_THRES - ep) / DECAY_THRES, 0.001)
        while True:
            action = agents.get_action(cur_state)[0]
            next_state, reward, done, info = env.step(action.astype(np.int32), unit='dBm')
            next_state = next_state.reshape((-1, env.n_states))
            agents.add_steps(cur_state, action, reward, done, next_state)
            loss = agents.learn()
            if loss:
                summary_writer.add_scalar('loss', loss, agents.eval_step)
            cur_state = next_state
            ep_his.append(reward/env.n_rx)
            if done:
                cum_reward = np.mean(ep_his)
                summary_writer.add_scalar('reward', cum_reward, ep)
                train_his.append({'cum_reward': cum_reward, 'ep_his': ep_his})
                if len(train_his) % 10 == 0:
                    print('EP: ', len(train_his),  'MADQN:',
                          np.mean([t['cum_reward'] for t in  train_his[-10:]]), info, flush=True)
                break
    print('calculating benckmarks')
    # find best ep_his
    train_his.sort(key=lambda o: o['cum_reward'], reverse=True)
    dqn_result = train_his[0]['cum_reward'], train_his[0]['ep_his']
    results = cal_benchmarks(env)
    result_path = logdir / 'results.log'
    with result_path.open('w') as f:
        for result in results:
            f.write(result[0] + ': ' + str(result[1]) + '\r\n')
        f.write('madqn: ' + str(dqn_result[0]) + '\r\n')
        f.write(str(dqn_result[1]))
    print('done')


def get_instance():
    env = utils.get_env()
    # for k, v in env.__dict__.items():
    #     print(f'{k}: {v}')
    n_states = env.n_states
    n_actions = env.n_actions
    agents = ListAgents(DQN, env.n_pair, n_states=n_states, n_actions=n_actions)
    logdir = utils.get_logdir()
    return env, agents, logdir


if __name__ == '__main__':
    from datetime import datetime
    start = datetime.now()
    instances = get_instance()
    rl_loop(*instances)
    end = datetime.now()
    print('cost time:', end - start)
