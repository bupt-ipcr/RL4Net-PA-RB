#!/usr/bin/env python
# coding=utf-8
"""
@author: Jiawei Wu
@create time: 2019-12-06 23:01
@edit time: 2020-12-28 17:22
@file: /PA/policy_ddpg.py
"""

import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import time
import gym
from vvlab.agents import DDPGBase
from vvlab.models import SimpleActorNet, SimpleCriticNet
CUDA = torch.cuda.is_available()


class CriticNet(nn.Module):
    """定义Critic的网络结构"""

    def __init__(self, n_states, n_actions, n_neurons=64):
        """
        定义隐藏层和输出层参数
        @param n_obs: number of observations
        @param n_actions: number of actions
        @param n_neurons: 隐藏层神经元数目
        """
        super(CriticNet, self).__init__()
        self.seq = nn.Sequential(
            nn.Linear(n_states+n_actions, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, s, a):
        """
        定义网络结构:
        state -> 全连接   -·-->  中间层 -> 全连接 -> ReLU -> Q值
        action -> 全连接  /相加，偏置
        """
        x = torch.cat((s, a), dim=-1)
        x = x.cuda() if CUDA else x
        q_value = self.seq(x)
        return q_value


class ActorNet(nn.Module):
    """定义Actor的网络结构"""

    def __init__(self, n_states, n_actions, n_neurons=30, a_bound=1):
        """
        定义隐藏层和输出层参数
        @param n_obs: number of observations
        @param n_actions: number of actions
        @param n_neurons: 隐藏层神经元数目
        @param a_bound: action的倍率
        """
        super(ActorNet, self).__init__()
        self.bound = a_bound
        self.seq = nn.Sequential(
            nn.Linear(n_states, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(64),
            nn.Linear(64, n_actions),
            nn.Sigmoid()
            # nn.Tanh()
        )

        if CUDA:
            self.bound = torch.FloatTensor([self.bound]).cuda()
        else:
            self.bound = torch.FloatTensor([self.bound])

    def forward(self, x):
        """
        定义网络结构: 第一层网络->ReLU激活->输出层->tanh激活->softmax->输出
        """
        x = x.cuda() if CUDA else x
        action_value = self.seq(x)
        # action_value = action_value * self.bound
        return action_value


class DDPG(DDPGBase):
    """DDPG类创建示例"""

    def _build_net(self):
        n_states, n_actions = self.n_states, self.n_actions
        self.actor_eval = ActorNet(
            n_states, n_actions, a_bound=self.bound)
        self.actor_target = ActorNet(
            n_states, n_actions, a_bound=self.bound)
        self.critic_eval = CriticNet(n_states, n_actions)
        self.critic_target = CriticNet(n_states, n_actions)

    def _build_noise(self):
        # self.noise = OUProcess(self.n_actions, sigma=0.1)
        # 当不需要nosie函数的特殊情况，可以略过
        pass

    def get_action(self, s):
        a = self._get_action(s)
        # a = 1 / (1 + np.exp(-a + 20))
        # a *= self.action_bound
        return a

    def get_action_noise(self, state, rate=1):
        action = self.get_action(state)
        # action_noise = np.random.normal(0, 2, size=(1, action.shape[1])) * rate
        # # action的后处理
        # action = action.squeeze()
        # action = self.action_bound * np.tanh(action/20)
        # for i, a in enumerate(action):
        #     action[i][action[i]!=np.max(a)] = 0
        # 使用均匀分布
        action_noise = np.random.uniform(-self.action_bound, self.action_bound, size=action.shape)  * rate

        action = np.clip(action + action_noise, 0, self.action_bound)
        action[action!=np.max(action)] = 0
        return action

    def add_step(self, s, a, r, d, s_):
        self._add_step(s, a, r, d, s_)

    def add_steps(self, cur_state, action, reward, done, next_state):
        size = action.shape[0]
        for i in range(size):
            self.add_step(cur_state[i], action[i], reward[i], done, next_state[i])
