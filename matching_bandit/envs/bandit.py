import gym
from gym import logger
from gym.utils import seeding

import matplotlib
import matplotlib.pyplot as plt

import numpy as np

class BanditEnv(gym.Env):
    """
    Abstract class for the bandit problem enviroment
    """

    def __init__(self, time_series_frequency):
        super(BanditEnv, self).__init__()
        self.time_series_frequency = time_series_frequency
        self.cum_regret = 0
        self.cum_regret_time_series = []
        self.steps = 0
        self.fig = None
    
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self, scenario_name):
        self.scenario_name = scenario_name
        self.cum_regret = 0
        self.cum_regret_time_series = []
        self.steps = 0

    #############################################################
    ###################### Logging regrets ######################
    #############################################################

    def update(self, instant_regret):
        self.cum_regret += instant_regret
        self.steps += 1
        # Update the cum_regret time series (for rendering)
        if self.steps % self.time_series_frequency == 0:
            self.cum_regret_time_series.append(self.cum_regret)
    
    #############################################################
    ######################### Plotting ##########################
    #############################################################
    
    def plot_regret(self):
        x = [i for i, _ in enumerate(self.cum_regret_time_series)]
        y = self.cum_regret_time_series
        plt.xticks(x, [(i + 1) * self.time_series_frequency for i, _ in enumerate(x)])
        plt.ylabel("cumulative regret")
        plt.xlabel("steps")
        plt.plot(x, y, marker='o')

    def log_regret(self):
        logger.info('Step: {}, Cumulative Regret: {}'.format(self.steps, self.cum_regret))

    def plot_utils(self, freeze=None, output_file=None):
        plt.tight_layout()

        if output_file is not None:
            self.fig.savefig(output_file)
        
        if freeze:
            # Keep the plot window open
            # https://stackoverflow.com/questions/13975756/keep-a-figure-on-hold-after-running-a-script
            if matplotlib.is_interactive(): 
                plt.ioff()
            plt.show(block=True)
        else:
            plt.show(block=False)
            plt.pause(0.001)
    
    def close(self):
        plt.close()