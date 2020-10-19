import utils
import numpy as np
from pa_dqn import DQN
from torch.utils.tensorboard import SummaryWriter
MAX_EPISODES = 1000


def rl_loop(env, agent):
    logdir = utils.get_logdir()
    summary_writer = SummaryWriter(log_dir=logdir)
    # train
    train_his = []
    for ep in range(MAX_EPISODES):
        cur_state = env.reset()
        cur_state = cur_state.reshape((-1, env.n_states))
        done = False
        while True:
            cum_reward = 0
            action = agent.get_action(cur_state)[0]
            next_state, reward, done, info = env.step(action.astype(np.int32))
            next_state = next_state.reshape((-1, env.n_states))
            agent.add_steps(cur_state, action, reward, done, next_state)
            loss = agent.learn()
            if loss:
                summary_writer.add_scalar('loss', loss, agent.eval_step)
            cur_state = next_state
            cum_reward += reward/env.n_recvs
            if done:
                summary_writer.add_scalar('reward', cum_reward, ep)
                train_his.append(cum_reward)
                if len(train_his) % 10 == 0:
                    print('EP: ', len(train_his),  'DQN:',
                          np.mean(train_his[-10:]), flush=True)
                break
    print('done')


def get_instance():
    env = utils.get_env()
    for k, v in env.__dict__.items():
        print(f'{k}: {v}')
    n_states = env.n_states
    n_actions = env.n_actions
    agent = DQN(n_states, n_actions)
    return env, agent


if __name__ == '__main__':
    from datetime import datetime
    start = datetime.now()
    env, agent = get_instance()
    rl_loop(env, agent)
    end = datetime.now()
    print('cost time:', end - start)
