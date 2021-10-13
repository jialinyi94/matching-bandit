import argparse
import numpy as np
import time

import gym
from gym import logger

import matplotlib
import matplotlib.pyplot as plt


class Experiment(object):
    """
    Repeated simulations to compare the averaged performance of agents on the same environment
    """
    def __init__(self, filename, num_reps, horizon):
        self.filename = filename
        self.title = str(num_reps) + ' repetitions'
        self.num_reps = num_reps
        self.horizon = horizon
        self.average_cumreg = {}
        self.max_cumreg = {}
        self.min_cumreg = {}
        self.ticks = []
        self.last_round = {}
    
    def run(self, env, agents):
        l = int(self.horizon / env.time_series_frequency)

        for i in range(l):
            tick = (i+1) * env.time_series_frequency
            self.ticks.append(tick)

        for agent in agents:
            multiworld_cumreg = []

            for j in range(self.num_reps):
                logger.info('========== Repetition: '+str(j+1)+' ==========')
                env.reset(agent.name + ': ' + str(j))
                agent.run(
                    env=env,
                    horizon=self.horizon,
                    animated=False
                )
                env.close()
                cumreg = env.cum_regret_time_series
                multiworld_cumreg.append(cumreg)
                agent.reset()

            self.average_cumreg[agent.name] = np.mean(multiworld_cumreg, axis=0)
            self.max_cumreg[agent.name] = np.percentile(multiworld_cumreg, q=95, axis=0)
            self.min_cumreg[agent.name] = np.percentile(multiworld_cumreg, q=5, axis=0)

            avg = self.average_cumreg[agent.name][-1]
            maximum = self.max_cumreg[agent.name][-1]
            minimum = self.min_cumreg[agent.name][-1]
            self.last_round[agent.name] = {'average': avg, 
                                           'max': maximum,
                                           'min': minimum}
        
    def plot(self, save_fig=False):
        matplotlib.rcParams['toolbar'] = 'toolbar2'

        fig = plt.figure(self.title, figsize=(10, 5))
        ax=fig.add_subplot(111)
        for agent_name in self.average_cumreg:
            ax.plot(self.ticks, 
                    self.average_cumreg[agent_name], 
                    marker='o', 
                    label=agent_name)
            ax.fill_between(self.ticks, 
                            self.min_cumreg[agent_name],
                            self.max_cumreg[agent_name],
                            alpha=0.2)
        plt.legend(loc=2) # lower right
        plt.ylabel("cumulative regret", fontsize=14)
        plt.xlabel("steps", fontsize=14)
        plt.show()
        if save_fig:
            fig.savefig('../figures/' + self.filename +'.png')
    
    def print_lastround(self):
        for agent in self.last_round:
            agent_info = "The cumlative regret of " + agent 
            regret_info = "in {0} steps: the average is {1}, 90% CI is [{2}, {3}]".format(self.horizon, 
                            int(self.last_round[agent]['average']), 
                            int(self.last_round[agent]['min']),
                            int(self.last_round[agent]['max']))
            print(agent_info + ' ' + regret_info + '\n')
    
    def statistics(self):
        avg_dict = {'mean': self.average_cumreg,
                    'low': self.min_cumreg,
                    'high': self.max_cumreg}
        return avg_dict


if __name__ == '__main__':
    '''
    Comparison between Rank1Elim and Rank1ElimDoubling
    '''
    import matching_bandit
    from matching_bandit.agents.rank1elim import Rank1Elim
    from matching_bandit.agents.rank1elimdt import Rank1ElimDT
    from matching_bandit.agents.uts import UTS
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--K', type=int, default=64)
    parser.add_argument('--L', type=int, default=64) 
    parser.add_argument('--horizon', type=int, default=2000000)
    parser.add_argument('--reps', type=int, default=1)
    parser.add_argument('--p_u', type=float, default=0.9)
    parser.add_argument('--p_v', type=float, default=0.9)
    parser.add_argument('--delta_u', type=float, default=0.5)
    parser.add_argument('--delta_v', type=float, default=0.5)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--output_file', default=None)
    args = parser.parse_args()

    logger.set_level(logger.INFO)

    time_series_frequency = args.horizon // 10

    agent1 = Rank1ElimDT(args.K, args.L, args.horizon)    
    agent2 = Rank1Elim(args.K, args.L)
    agent3 = UTS(args.K, args.L, gamma=2)

    
    env = gym.make(
        'BipartitePairBandit-v0',
        nrows = args.K,
        ncols = args.L,
        time_series_frequency = time_series_frequency
    )
    #env.seed(args.seed)

    row_pars = (args.p_u - args.delta_u) * np.random.uniform(size=args.K) 
    row_pars[0] = args.p_u
    col_pars = (args.p_v - args.delta_v) * np.random.uniform(size=args.L) 
    col_pars[0] = args.p_v

    env.reset('start', row_dist=row_pars, col_dist=col_pars)


    simulation = Experiment('test', num_reps=args.reps, horizon=args.horizon)

    t = time.time()
    simulation.run(env, [agent3])
    escaped = time.time() - t
    simulation.print_lastround()
    simulation.plot()
    print('------------- Escaped time is {0} seconds -------------'.format(escaped))