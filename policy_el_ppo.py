#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@author: Jiawei Wu
@create time: 2021-03-30 15:11
@edit time: 2021-04-01 16:50
@file: /RL4Net-PA-RB/policy_el_ppo_contious.py
@desc: 
"""

import numpy as np
from policy_dqn import DQN as RLAgent
from elegantrl.run import Arguments
from elegantrl.agent import AgentPPO as ELAgent
from elegantrl.agent import ReplayBuffer
import warnings
import torch

CUDA = torch.cuda.is_available()


class DiscrateAgentAdapter(RLAgent, ELAgent):

    def _set_default(self, kwargs):
        """set default to kwargs( inplace since it's a dictionary)"""
        dft_net_dim = 16
        if 'net_dim' not in kwargs:
            msg = f"net_dim not in kwargs, set default to {dft_net_dim}"
            warnings.warn(msg)
        kwargs['net_dim'] = dft_net_dim

        dft_target_step = 1
        if 'target_step' not in kwargs:
            msg = f"target_step not in kwargs, set default to {dft_target_step}"
            warnings.warn(msg)
        kwargs['target_step'] = dft_target_step

        dft_if_discrete = False  # for PPO
        if 'if_discrete' not in kwargs:
            msg = f"if_discrete not in kwargs, set default to {dft_if_discrete}"
            warnings.warn(msg)
        kwargs['if_discrete'] = dft_if_discrete

    def __init__(self, n_states, n_actions, *args, **kwargs):
        """
        ELAgent.__init__(self)
        ELAgent.init(self, net_dim, state_dim, action_dim)
        """
        self._set_default(kwargs)

        # eval_step for display
        self.eval_step = 0

        self.if_discrete = kwargs['if_discrete']
        self.if_on_policy = True  # since PPO

        # default args for hyper-parameters
        self.hypers = Arguments(if_on_policy=self.if_on_policy)
        self.hypers.target_step = kwargs.get('target_step')
        self.hypers.net_dim = kwargs.get('net_dim')
        self.hypers.repeat_times = 4  # for PPO

        self.el_agent = ELAgent()
        action_dim = 1 if self.if_discrete else n_actions
        self.el_agent.init(self.hypers.net_dim, n_states, action_dim)

        self.el_buffer = ReplayBuffer(
            max_len=self.hypers.max_memo + self.hypers.target_step, if_gpu=CUDA,
            if_on_policy=self.if_on_policy, state_dim=n_states,
            action_dim=action_dim
        )

    @property
    def epsilon(self):
        return self.explore_rate

    @epsilon.setter
    def epsilon(self, value):
        self.explore_rate = value

    def add_steps(self, cur_state, action, reward, done, noise):
        """Use el_buffer to simulate agent.add_steps"""
        size = action.shape[0]
        for i in range(size):
            state = cur_state[i]
            mask = 0.0 if done else self.hypers.gamma   # same to el_buffer
            other = (reward[i], mask, *action[i], *noise[i])
            self.el_buffer.append_buffer(state, other)
        self.el_buffer.update_now_len_before_sample()

    def get_action(self, cur_state):
        # There are n_tx states in cur_state in PAEnv
        actions, noises = [], []
        for state in cur_state:
            action, noise = self.el_agent.select_action(state)
            actions.append(action)
            noises.append(noise)

        return np.array(actions), np.array(noises)

    def learn(self, **kwargs):
        """
        def update_net(self, buffer, target_step, batch_size, repeat_times) -> (float, float):
            ...
            return next_q.mean().item(), obj_critic.item() / 2
        """
        target_step = kwargs.get('target_step', self.hypers.target_step)
        batch_size = kwargs.get('batch_size', self.hypers.batch_size)
        repeat_times = kwargs.get('repeat_times', self.hypers.repeat_times)

        self.eval_step += 1

        if self.eval_step <= batch_size:
            return None

        q_value, loss = self.el_agent.update_net(
            self.el_buffer, target_step, batch_size, repeat_times)

        # clear buffer after train (or say, before next explore)
        self.el_buffer.empty_buffer_before_explore()

        return loss  # corresponding to rl4net.agent.DQNBase.learn
