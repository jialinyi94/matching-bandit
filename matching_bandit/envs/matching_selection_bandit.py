import gym
from gym import spaces
from gym.utils import seeding

import numpy as np

import matching_bandit
from matching_bandit.envs.bandit import BanditEnv
from gym import logger
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
matplotlib.rcParams['toolbar'] = 'None'

class MatchingSelectionBanditEnv(BanditEnv):
    """
    Description:

    Observation:
        "trials": symmetric matrix, [i,j] gives the trials of pair (i,j)
        "rewards": symmetric matrix, [i,j] gives the cumulative reward of pair (i,j)
        "feedback": n_pairs-array, [k] gives the instant random reward for pair (2k, 2k+1)
 
    Action:
        Type: MultiDiscrete([n_items]*n_items)

    Reward:
        Reward is a Bernoulli random variable with mean as a function (reward_type) of the parameters of 
        the 2 arms. 

        reward_type     function
        min             f(p, q) = pq
    """

    metadata = {
        'render.modes': ['human']
    }

    def __init__(self, n_pairs, time_series_frequency, reward_type='min'):
        if reward_type != 'min':
            raise NotImplementedError

        super(MatchingSelectionBanditEnv, self).__init__(time_series_frequency)
        self.n_pairs = n_pairs
        self.n_items = n_pairs*2
        self.arm_probs = np.random.uniform(size=self.n_items)
        self.reward_type = reward_type

        # gym environment
        self.action_space = spaces.MultiDiscrete([self.n_items]*self.n_items)
        self.observation_space = spaces.Dict({
            'trials': spaces.Box(low=0, high=np.inf, shape=(self.n_items,self.n_items)),
            'rewards': spaces.Box(low=0, high=np.inf, shape=(self.n_items,self.n_items)),
            'feedback': spaces.Box(low=0, high=1.0, shape=(self.n_pairs,))
        })
    

    def reset(self, scenario_name, item_dist=None):
        super(MatchingSelectionBanditEnv, self).reset(scenario_name)
        self.trials = np.zeros(shape=(self.n_items,self.n_items)) # number of trials
        self.rewards = np.zeros(shape=(self.n_items,self.n_items)) # cumulative rewards

        if item_dist is not None:
            self.arm_probs = item_dist
        logger.info('Bernoulli parameters: {}'.format(self.arm_probs))
        # optimal matching
        self.opt_m = np.argsort(self.arm_probs)
        self.opt_admatrix = np.zeros(shape=(self.n_items,self.n_items))
        self.opt_v = 0
        for k in range(self.n_pairs):
            i = self.opt_m[2*k]
            j = self.opt_m[2*k+1]
            self.opt_v += self.arm_probs[i] * self.arm_probs[j]
            self.opt_admatrix[i,j] = 1
            self.opt_admatrix[j,i] = 1

    def step(self, action):
        feedbacks = np.zeros(shape=(self.n_pairs,))
        exp_rwd = 0
        for k in range(self.n_pairs):
            i = action[2*k]
            j = action[2*k+1]
            # rank-1 assumption in the reward function
            success_prob = self.arm_probs[i] * self.arm_probs[j]
            exp_rwd += success_prob # expected reward
            random_rwd = np.random.binomial(1, success_prob) # stochastic reward
            feedbacks[k] = random_rwd

            self.trials[i,j] += 1
            self.trials[j,i] += 1
            self.rewards[i,j] += random_rwd
            self.rewards[j,i] += random_rwd

        instant_regret = self.opt_v - exp_rwd
        super(MatchingSelectionBanditEnv, self).update(instant_regret)
        
        obs = {
            'trials': self.trials,
            'rewards': self.rewards,
            'feedback': feedbacks
        }
        done = False
        reward = sum(feedbacks)
        info = {}
        return obs, reward, done, info
    
    def render(self, mode='human', freeze=None, output_file=None):
        if mode != 'human':
            raise NotImplementedError

        self.fig = plt.figure(self.scenario_name, figsize=(9, 6))
        grid_size = (7, 2)

        # ------- Plot cumulative regret time serie -------
        plt.subplot2grid(grid_size, (0,0), rowspan=3, colspan=2)
        plt.title('N = {}'.format(self.n_pairs))
        super(MatchingSelectionBanditEnv, self).plot_regret()

        # ------- Plot trials -------
        plt.subplot2grid(grid_size, (3, 0), rowspan=4, colspan=1)
        plt.title('trials')
        sns.heatmap(self.trials)

        # ------- Visualize the optimal matching -------
        plt.subplot2grid(grid_size, (3, 1), rowspan=4, colspan=1)
        plt.title('optimal matching')
        sns.heatmap(self.opt_admatrix)

        super(MatchingSelectionBanditEnv, self).plot_utils(
            freeze=freeze, output_file=output_file)


if __name__ == '__main__':
    '''
    Implement random policy on MatchingSelectionBanditEnv
    '''
    #from math import factorial

    N = 4
    T = 100000
    #env = MatchingSelectionBanditEnv(N, T // 10)
    env = gym.make(
        'MatchingSelectionBandit-v0',
        n_pairs=N,
        time_series_frequency = T // 10
    )
    # test generating matchings
    '''
    matching_set = env.all_matchings()
    expected = factorial(2*N) / factorial(N) / 2**N
    assert len(matching_set) == expected
    print(matching_set)
    '''
    # test reset(), step(), render()
    # probs = [0.1, 0.4, 0.3, 0.2, 0.6, 0.5]
    env.reset('test')
    for i in range(T):
        action =  np.random.permutation(env.n_items)
        env.step(action)
        if (i+1) % env.time_series_frequency == 0:
            env.log_regret()
            env.render()





        

