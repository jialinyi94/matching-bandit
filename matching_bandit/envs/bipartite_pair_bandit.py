import gym
from gym import spaces
from gym.utils import seeding

import numpy as np

import matching_bandit
from matching_bandit.envs.bandit import BanditEnv

import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['toolbar'] = 'None'

class BipartitePairBanditEnv(BanditEnv):
    """
    Description:

    Observation:
        Type: Dict {
            'trials': Box(low=0, high=np.inf, shape=(nrows * ncols,))
            'rewards': Box(low=0, high=np.inf, shape=(nrows * ncols,))
        }
        Observation       Index                Pair       Min      Max
        'trial'           i * ncols + j        (i,j)       0       Inf
        'rewards'         i * ncols + j        (i,j)       0       Inf

    Action:
        Type: Discrete(nrows * ncols)

        Num                             Observation: pair
        0                               (0, 0)
        i * ncols + j                   (i, j)

    Reward:
        Reward is a Bernoulli random variable with mean as a function (reward_type) of the parameters of 
        the 2 arms. 

        reward_type     function
        min             f(p, q) = pq

    """

    metadata = {
        'render.modes': ['human']
    }

    def __init__(self, nrows, ncols, time_series_frequency, reward_type='min'):
        if reward_type != 'min':
            raise NotImplementedError

        super(BipartitePairBanditEnv, self).__init__(time_series_frequency)
        self.nrows = nrows
        self.ncols = ncols
        self.num_pairs = nrows * ncols
        self.arm_probs = {
            'row': np.random.uniform(size=nrows),
            'col': np.random.uniform(size=ncols)
        }
        self.reward_type = reward_type

        # Initial state (can be reset later)
        self.trials = {
            'row': [0]*self.nrows,
            'col': [0]*self.ncols
        }
        self.states = {
            'trials': np.array([0] * self.num_pairs),
            'rewards': np.array([0] * self.num_pairs)
        }

        # gym environment
        self.action_space = spaces.Discrete(self.num_pairs)
        self.observation_space = spaces.Dict({
            'trials': spaces.Box(low=0, high=np.inf, shape=(self.num_pairs,)),
            'rewards': spaces.Box(low=0, high=np.inf, shape=(self.num_pairs,))
        })
    
    def arm_ids(self, action):
        row_arm = action // self.ncols
        col_arm = action % self.ncols
        return row_arm, col_arm

    def step(self, action):
        # 1) Generating the reward
        row_arm, col_arm = self.arm_ids(action)
        if self.reward_type == 'min':
            # rank-1 assumption in the reward function
            success_prob = self.arm_probs['row'][row_arm] * self.arm_probs['col'][col_arm]
        reward = np.random.binomial(1, success_prob)

        # 2) Updating the observation states (pairs)
        self.states['trials'][action] += 1
        self.states['rewards'][action] += reward

        # 3) Updating the states for arms
        self.trials['row'][row_arm] += 1
        self.trials['col'][col_arm] += 1

        # 4) Recording the cumulative regret
        instant_regret = self.opt_v - success_prob
        super(BipartitePairBanditEnv, self).update(instant_regret)

        # observations
        obs = self.states

        # Regret minimisation: no stopping rule
        done = False
    
        # Optionally we can pass additional info, we are not using that for now
        info = {}

        return obs, reward, done, info

    def reset(self, scenario_name, row_dist=None, col_dist=None):
        super(BipartitePairBanditEnv, self).reset(scenario_name)
        self.trials = {
            'row': [0]*self.nrows,
            'col': [0]*self.ncols
        }
        self.states = {
            'trials': np.array([0] * self.num_pairs),
            'rewards': np.array([0] * self.num_pairs)
        }

        if (row_dist is not None) and (col_dist is not None):
            self.arm_probs['row'] = row_dist
            self.arm_probs['col'] = col_dist

        # optimal pair
        if self.reward_type == 'min':
            opt_row_arm = np.argsort(self.arm_probs['row'])[-1]
            opt_col_arm = np.argsort(self.arm_probs['col'])[-1]
            self.opt_p = (opt_row_arm, opt_col_arm)
            self.opt_v = self.arm_probs['row'][opt_row_arm] * self.arm_probs['col'][opt_col_arm]

        obs = self.states

        return obs

    def render(self, mode='human', freeze=None, output_file=None):
        if mode != 'human':
            raise NotImplementedError

        self.fig = plt.figure(self.scenario_name, figsize=(9, 6))
        grid_size = (7, 2)

        # ------- Plot cumulative regret time serie -------
        plt.subplot2grid(grid_size, (0,0), rowspan=3, colspan=2)
        plt.title('K = {}, L = {}'.format(self.nrows, self.ncols))
        super(BipartitePairBanditEnv, self).plot_regret()

        # ------- Plot trials -------
        # row arms
        plt.subplot2grid(grid_size, (3, 0), rowspan=2, colspan=1)
        x = list(range(self.nrows))
        plt.barh(x, self.trials['row'])
        plt.ylabel('rows')
        plt.xlabel('trials')
        plt.yticks(x, ['' for i in x])

        # col arms
        plt.subplot2grid(grid_size, (5, 0), rowspan=2, colspan=1)
        x = list(range(self.ncols))
        plt.barh(x, self.trials['col'])
        plt.ylabel('columns')
        plt.xlabel('trials')
        plt.yticks(x, ['' for i in x])

        # ------- Plot true probabilities -------
        # row arms
        plt.subplot2grid(grid_size, (3, 1), rowspan=2, colspan=1)
        x = list(range(self.nrows))
        plt.barh(x, self.arm_probs['row'])
        plt.ylabel('rows')
        plt.xlabel('true parameter')
        plt.yticks(x, ['' for i in x])

        # col arms
        plt.subplot2grid(grid_size, (5, 1), rowspan=2, colspan=1)
        x = list(range(self.ncols))
        plt.barh(x, self.arm_probs['col'])
        plt.ylabel('columns')
        plt.xlabel('true parameter')
        plt.yticks(x, ['' for i in x])

        super(BipartitePairBanditEnv, self).plot_utils(
            freeze=freeze, output_file=output_file)
