#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@author: Jiawei Wu
@create time: 1970-01-01 08:00
@edit time: 2020-12-31 10:00
@file: /PA/auto_run_jsac.py
@desc: 
"""
import utils
from pa_main import rl_loop


if __name__ == '__main__':
    args = utils.get_args()
    print(args)
    seeds = utils.create_seeds()
    # iter seeds
    for seed in seeds[args.offset:args.offset+args.seeds]:
        args.env.update({'seed': seed})
        rl_loop(args)
