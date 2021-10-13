import argparse
import sys
import time
import math

import numpy as np

import gym
from gym import logger

import matching_bandit

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', default='MonopartitePairBandit-v0')
    parser.add_argument('--num_arms', type=int, default=10)
    parser.add_argument('--num_left_arms', type=int, default=10)
    parser.add_argument('--num_right_arms', type=int, default=10)
    parser.add_argument('--horizon', type=int, default=10000)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--output_file', default=None)
    args = parser.parse_args()

    logger.set_level(logger.INFO)

    time_series_frequency = args.horizon // 10

    # Set up the environment
    if args.env == 'BipartitePairBandit-v0':
        env = gym.make(
                    args.env, 
                    num_left_arms = args.num_left_arms,
                    num_right_arms = args.num_right_arms,
                    time_series_frequency = time_series_frequency
        )
    elif args.env == 'MonopartitePairBandit-v0':
        env = gym.make(
                    args.env, 
                    num_arms = args.num_arms,
                    time_series_frequency = time_series_frequency
        )
    else:
        raise NotImplementedError
    
    env.seed(args.seed)

    # Simulation loop
    env.reset('Random Agent')
    for i in range(args.horizon):
        # Action/Feedback
        action = env.action_space.sample()
        env.step(action)

        # Render the current state
        if (i+1) % time_series_frequency == 0:
            env.render()
        
    # Render the final state and keep the plot window open
    env.render(freeze=True, output_file=args.output_file)

    env.close()