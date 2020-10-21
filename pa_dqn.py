#!/usr/bin/env python
# coding=utf-8
"""
@author: ,: Jiawei Wu
@create time: 2019-12-07 20:17
@edit time: ,: 2020-10-20 19:05
@FilePath: ,: /PA/pa_dqn.py
"""
from functools import reduce
import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from vvlab.agents import DQNBase


CUDA = torch.cuda.is_available()


class PADQNNet(nn.Module):
    """一个只有一层隐藏层的DQN神经网络"""

    def __init__(self, n_states, n_actions, card_no=0):
        """
        定义隐藏层和输出层参数
        @param n_obs: number of observations
        @param n_actions: number of actions
        @param n_neurons: number of neurons for the hidden layer
        """
        super(PADQNNet, self).__init__()
        self.card_no = card_no
        self.seq = nn.Sequential(
            nn.Linear(n_states, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, n_actions),
            nn.ReLU(),
        )

    def forward(self, x):
        """
        定义网络结构: 第一层网络->ReLU激活->输出层->softmax->输出
        """
        if CUDA:
            x = x.cuda(self.card_no)
        action_values = self.seq(x)
        return action_values


class DQN(DQNBase):
    """
    基于DQNBase创建的DQN类，通过附带的简单神经网络创建了 eval dqn network 和 target dqn network
    """

    def __init__(self, n_states, n_actions, **kwargs):
        super().__init__(n_states, n_actions, **kwargs)

    def _build_net(self):
        self.eval_net = PADQNNet(self.n_states, self.n_actions, self.card_no)
        self.target_net = PADQNNet(self.n_states, self.n_actions, self.card_no)

    def add_steps(self, cur_state, action, reward, done, next_state):
        size = action.shape[0]
        for i in range(size):
            self.add_step(cur_state[i], action[i], reward, done, next_state[i])

    def get_action(self, state):
        # 将行向量转为列向量（1 x n_states -> n_states x 1 x 1)
        if np.random.rand() < self.epsilon:
            # 概率随机
            action_size = state.shape[0]
            return np.random.randint(0, self.n_actions, (1, action_size))
        else:
            # greedy
            state = torch.FloatTensor(state)
            state = torch.unsqueeze(torch.FloatTensor(state), 0)
            action_values = self.eval_net.forward(state).cpu()
            return action_values.data.numpy().argmax(axis=2)
