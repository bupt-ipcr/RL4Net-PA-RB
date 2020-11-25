'''
Author: your name
Date: 1970-01-01 08:00:00
LastEditTime: 2020-11-23 09:38:13
LastEditors: your name
Description: In User Settings Edit
FilePath: /PA/ddpg_main.py
'''

import utils
import numpy as np
from pa_dqn import DQN
from pa_ddpg import DDPG
from torch.utils.tensorboard import SummaryWriter
from benckmarks import cal_benchmarks
MAX_EPISODES = 1000
DECAY_THRES = 400


def rl_loop(env, agent, logdir):
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
            action = agent.get_action_noise(cur_state, rate=rate).squeeze()
            next_state, reward, done, info = env.step(action.astype(np.float32), raw=True)
            next_state = next_state.reshape((-1, env.n_states))
            agent.add_steps(cur_state, action, reward/env.n_recvs, done, next_state)
            losses = agent.learn()
            if losses:
                summary_writer.add_scalar('c_loss', losses[0], agent.step)
                summary_writer.add_scalar('a_loss', losses[1], agent.step)
            cur_state = next_state
            ep_his.append(reward/env.n_recvs)
            if done:
                cum_reward = np.mean(ep_his)
                summary_writer.add_scalar('reward', cum_reward, ep)
                train_his.append({'cum_reward': cum_reward, 'ep_his': ep_his})
                if len(train_his) % 10 == 0:
                    print('EP: ', len(train_his),  'DDPG:',
                          np.mean([t['cum_reward'] for t in  train_his[-10:]]), flush=True)
                    print(action)
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
        f.write('ddpg: ' + str(dqn_result[0]) + '\r\n')
        f.write(str(dqn_result[1]))
    print('done')


def get_instance():
    env = utils.get_env(policy='ddpg')
    for k, v in env.__dict__.items():
        print(f'{k}: {v}')
    n_states = env.n_states
    n_actions = env.n_actions
    agent = DDPG(n_states, 1, bound=6.3, buff_size=50000, buff_thres=4000, batch_size=32, lr_a=0.0001, lr_c=0.001, tau=0.01, gamma=0.0)
    logdir = utils.get_logdir()
    return env, agent, logdir


if __name__ == '__main__':
    from datetime import datetime
    start = datetime.now()
    instances = get_instance()
    rl_loop(*instances)
    end = datetime.now()
    print('cost time:', end - start)
