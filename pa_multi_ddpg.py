#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@author: Jiawei Wu
@create time: 1970-01-01 08:00
@edit time: 2020-12-28 15:41
@file: /PA/pa_multi_ddpg.py
@desc: 
"""
import utils
import numpy as np
from pa_ddpg import DDPG
from torch.utils.tensorboard import SummaryWriter
from benckmarks import cal_benchmarks

MAX_EPISODES = 1000
DECAY_THRES = 400


class ListAgents:
    def __init__(self, policy, num, *args, **kwargs):
        self._step = 0
        self.agents = [policy(*args, **kwargs) for _ in range(num)]

    def get_action_noise(self, states, *args, **kwargs):
        action = np.array([
            agent.get_action_noise(np.array([state]), *args, **kwargs) for state, agent in zip(states, self.agents)
        ])
        return action.T

    def add_steps(self, cur_states, actions, rewards, done, next_states):
        for cur_state, action, reward, next_state, agent in zip(cur_states, actions, rewards, next_states, self.agents):
            agent.add_step(cur_state, action, reward, done, next_state)
    # def add_steps(self, cur_states, actions, reward, done, next_states):
    #     for cur_state, action, next_state, agent in zip(cur_states, actions, next_states, self.agents):
    #         agent.add_step(cur_state, action, reward, done, next_state)

    def learn(self):
        self._step += 1
        losses = []
        for agent in self.agents:
            loss = agent.learn()
            if loss:
                losses.append(loss)
        return np.mean(losses, axis=0) if losses else None

    @property
    def eval_step(self):
        return self._step



@utils.timeit
def maddpg_loop(env, agents, logdir):
    summary_writer = SummaryWriter(log_dir=logdir)
    # train
    train_his = []
    for ep in range(MAX_EPISODES):
        cur_state = env.reset()
        cur_state = cur_state.reshape((-1, env.n_states))
        done = False
        ep_his = []
        # rate = max((DECAY_THRES - ep) / DECAY_THRES, 0.01)
        # 使用EP的倒数作为rate
        rate = 1. / (ep + 1)
        while True:
            action = agents.get_action_noise(cur_state, rate=rate).squeeze()
            next_state, reward, done, info = env.step(
                action.astype(np.float32), unit='dBm')
            next_state = next_state.reshape((-1, env.n_states))
            agents.add_steps(cur_state, action, reward, done, next_state)
            losses = agents.learn()
            if losses is not None:
                summary_writer.add_scalar('c_loss', losses[0], agents.eval_step)
                summary_writer.add_scalar('a_loss', losses[1], agents.eval_step)
            cur_state = next_state
            ep_his.append(info['rate'])
            if done:
                cum_reward = np.mean(ep_his)
                summary_writer.add_scalar('reward', cum_reward, ep)
                train_his.append({'cum_reward': cum_reward, 'ep_his': ep_his})
                if len(train_his) % 10 == 0:
                    print('EP: ', len(train_his),  'DDPG:',
                          np.mean([t['cum_reward'] for t in train_his[-10:]]), info, flush=True)
                break

    # find best ep_his
    train_his.sort(key=lambda o: o['cum_reward'], reverse=True)
    dqn_result = train_his[0]['cum_reward'], train_his[0]['ep_his']
    return dqn_result


def get_maddpg_agents(env, **kwargs):
    n_states = env.n_states
    n_valid_rb = env.n_valid_rb
    agents = ListAgents(DDPG, env.n_pair, n_states=n_states,
                        n_actions=n_valid_rb, bound=38, buff_size=50000,
                        buff_thres=64, lr_a=0.0001, lr_c=0.001, tau=0.01, gamma=0.1, **kwargs)
    return agents


def get_instances(args=utils.get_args()):
    env = utils.get_env(**args.env)
    agent = get_maddpg_agents(env, **args.agent)
    conf = utils.get_config('config.yaml')
    conf['env'].update(args.env)
    conf['agent'].update(args.agent)
    logdir = utils.get_logdir(conf)
    return env, agent, logdir


def demo(env, agent, logdir):
    ddpg_result = ddpg_loop(env, agent, logdir)

    result_path = logdir / 'results.log'
    with result_path.open('w') as f:
        # RL results
        f.write('dqn: ' + str(ddpg_result[0]) + '\r\n')
        # f.write(str(dqn_result[1]))
        # benckmarks
        results = cal_benchmarks(env)
        for result in results:
            f.write(result[0] + ': ' + str(result[1]) + '\r\n')
    print('done')


def demo(env, agent, logdir):
    madqn_result = maddpg_loop(env, agent, logdir)

    result_path = logdir / 'results.log'
    with result_path.open('w') as f:
        # RL results
        f.write('maddpg: ' + str(madqn_result[0]) + '\r\n')
        # benckmarks
        results = cal_benchmarks(env)
        for result in results:
            f.write(result[0] + ': ' + str(result[1]) + '\r\n')
    print('done')


if __name__ == '__main__':
    instances = get_instances()
    demo(*instances)
