import gym
from gym import spaces
from gym.utils import seeding

import numpy as np
import matching_bandit
from matching_bandit.envs.bandit import BanditEnv

import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['toolbar'] = 'None'

class MonopartitePairBanditEnv(BanditEnv):
    """
    Description:
        The pair bandit is inherited from the vanilla multi-armed bandit enviroment.
        Each round, the agent chooses a pair of arms and only observes the feedback from
        this pair and receives a reward from it.
        The goal of the agent is to maximise the cumulative reward over a fixed horizon.
    
    Action:
        Type: Discrete(num_arms * (num_arms-1) / 2)

        Num                                      Observation: pair
        0                                        (1, 0)
        1 2                                      (2, 0) (2, 1)
        ......                                   ......

        (i-1)*i / 2 + j                          (i, j)

    Observation:
        Type: Dict {
            "pair": Discrete(num_arms * (num_arms-1) / 2)
            "reward": Box(low=0.0, high=1.0, shape=(1,))
        }

    Reward:
        Reward is a Bernoulli random variable with mean as a function (reward_type) of the parameters of 
        the 2 arms. 

        reward_type     function
        min             f(p, q) = pq

    """
    metadata = {
        'render.modes': ['human']
    }

    def __init__(self, num_arms, time_series_frequency, reward_type='min'):
        if num_arms %2 != 0 or num_arms < 2:
            raise ValueError("The number of arms should be a even number >= 2")

        if reward_type != 'min':
            raise NotImplementedError

        super(MonopartitePairBanditEnv, self).__init__(time_series_frequency)
        # Success probablities of arms
        self.num_arms = num_arms
        self.num_pairs = num_arms * (num_arms - 1) // 2
        self.arm_probs = np.random.uniform(size=num_arms)
        self.reward_type = reward_type

        # Initial state (can be reset later)
        self.trials = [0]*self.num_arms
        self.states = {
            'trials': np.array([0] * self.num_pairs),
            'rewards': np.array([0] * self.num_pairs)
        }

        # gym environment
        self.action_space = spaces.Discrete(self.num_pairs)
        # Observation is a tuple of (action, reward)
        self.observation_space = spaces.Dict({
            'trials': spaces.Box(low=0, high=np.inf, shape=(self.num_pairs,)),
            'rewards': spaces.Box(low=0, high=np.inf, shape=(self.num_pairs,))
        })

    def arm_ids(self, action):
        if action >= self.num_pairs:
            raise ValueError('Action not defined')
        counts = 1
        left_arm = 1
        while action >= counts:
            left_arm += 1
            counts += left_arm
        right_arm = action - left_arm*(left_arm-1) // 2
        return left_arm, right_arm

    def step(self, action):
        # 1) Generating the reward
        left_arm, right_arm = self.arm_ids(action)
        # rank-1 assumption in the reward function
        if self.reward_type == 'min':
            success_prob = self.arm_probs[left_arm] * self.arm_probs[right_arm]
        reward = np.random.binomial(1, success_prob)

        # 2) Updating the observation states (pairs)
        self.states['trials'][action] += 1
        self.states['rewards'][action] += reward
        
        # 3) Updating the states for arms
        self.trials[left_arm] += 1
        self.trials[right_arm] += 1
        
        # 4) Recording the cumulative regret
        instant_regret = self.opt_v - success_prob
        super(MonopartitePairBanditEnv, self).update(instant_regret)

        # observations
        obs = self.states

        # Regret minimisation: no stopping rule
        done = False
    
        # Optionally we can pass additional info, we are not using that for now
        info = {}

        return obs, reward, done, info

    def reset(self, scenario_name, p_dist=None):
        super(MonopartitePairBanditEnv, self).reset(scenario_name)
        self.trials = [0]*self.num_arms
        self.states = {
            'trials': np.array([0] * self.num_pairs),
            'rewards': np.array([0] * self.num_pairs)
        }

        # reset the probability parameters
        if p_dist is not None:
            if np.min(p_dist) < 0 or np.max(p_dist) > 1:
                raise ValueError("All probabilities must be between 0 and 1.")
            if len(p_dist) != self.num_arms:
                raise ValueError("The length of list not equal to the number of arms.")
            self.arm_probs = p_dist
        
        # optimal pair
        if self.reward_type == 'min':
            self.opt_p = np.argsort(self.arm_probs)[-2:]
            probs_s = sorted(self.arm_probs)
            self.opt_v = probs_s[-1] * probs_s[-2]
        
        obs = self.states

        return obs

    def render(self, mode='human', freeze=None, output_file=None):
        if mode != 'human':
            raise NotImplementedError

        self.fig = plt.figure(self.scenario_name, figsize=(9, 6))
        grid_size = (5, 2)

        # ------- Plot cumulative regret time serie -------
        plt.subplot2grid(grid_size, (0,0), rowspan=3, colspan=2)
        plt.title('# arms = {}'.format(self.num_arms))
        super(MonopartitePairBanditEnv, self).plot_regret()
        
        # ------- Plot trials -------
        plt.subplot2grid(grid_size, (3, 0), rowspan=2, colspan=1)
        x = list(range(self.num_arms))
        plt.barh(x, self.trials)
        plt.ylabel('arms')
        plt.xlabel('trials')
        plt.yticks(x, ['' for i in x])

        # ------- Plot true probabilities -------
        plt.subplot2grid(grid_size, (3, 1), rowspan=3, colspan=1)
        plt.barh(x, self.arm_probs)
        plt.ylabel('arms')
        plt.xlabel('true parameter')
        plt.yticks(x, ['' for i in x])

        super(MonopartitePairBanditEnv, self).plot_utils(
            freeze=freeze, output_file=output_file)