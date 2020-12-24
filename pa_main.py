#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@author: Jiawei Wu
@create time: 1970-01-01 08:00
@edit time: 2020-12-23 15:30
@FilePath: /PA/pa_main.py
@desc: 
"""
import utils
from benckmarks import cal_benchmarks
from pa_dqn import dqn_loop, get_dqn_agent
from pa_multi_dqn import madqn_loop, get_madqn_agents

MAX_EPISODES = 1000
DECAY_THRES = 400


@utils.timeit
def rl_loop():
    env = utils.get_env()
    logdir = utils.get_logdir()

    madqn_agents = get_madqn_agents(env)
    madqn_result = madqn_loop(env, madqn_agents, logdir)

    dqn_agent = get_dqn_agent(env)
    dqn_result = dqn_loop(env, dqn_agent, logdir)

    result_path = logdir / 'results.log'
    with result_path.open('w') as f:
        # RL results
        f.write('madqn: ' + str(madqn_result[0]) + '\r\n')
        # f.write(str(madqn_result[1]))
        f.write('dqn: ' + str(dqn_result[0]) + '\r\n')
        # f.write(str(dqn_result[1]))
        # benckmarks
        results = cal_benchmarks(env)
        for result in results:
            f.write(result[0] + ': ' + str(result[1]) + '\r\n')
    print('done')


if __name__ == '__main__':
    rl_loop()