#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@author: Jiawei Wu
@create time: 1970-01-01 08:00
@edit time: 2020-12-31 09:56
@file: /PA/pa_main.py
@desc: 
"""
import utils
from benckmarks import cal_benchmarks
from pa_dqn import dqn_loop, get_dqn_agent
from pa_multi_dqn import madqn_loop, get_madqn_agents

MAX_EPISODES = 1000
DECAY_THRES = 400




@utils.timeit
def rl_loop(args=utils.get_args()):
    env = utils.get_env(**args.env)
    conf = utils.get_config('config.yaml')
    conf['env'].update(args.env)
    conf['agent'].update(args.agent)
    logdir = utils.get_logdir(conf)

    if args.ignore and utils.check_exist(logdir):
        print(f"Ingore seed {conf['env']['seed']}!")
        return
    madqn_agents = get_madqn_agents(env, **args.agent)
    madqn_result = madqn_loop(env, madqn_agents, logdir)

    dqn_agent = get_dqn_agent(env, **args.agent)
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
    print('Done!')


if __name__ == '__main__':
    rl_loop()