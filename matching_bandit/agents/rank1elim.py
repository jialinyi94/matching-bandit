import argparse

import numpy as np

import gym
from gym import logger

import matching_bandit


class Rank1Elim(object):
    """
    Implementation of Rank1Elim algorithm in  Katariya et al (2017)
    """
    def __init__(self, K=10, L=10):
        self.name = 'Rank1Elim'
        self.K = K
        self.L = L
        self.t = 1
        self.delta_l = 1
        self.CU = np.zeros(shape=(K,L))
        self.CV = np.zeros(shape=(K,L))
        self.hU = list(range(K))
        self.hV = list(range(L))
        self.l=0
        self.n_lminus1 = 0
        self.UCB = {
            'rows': np.zeros(shape=(self.K,)),
            'columns': np.zeros(shape=(self.L))
        }
        self.LCB = {
            'rows': np.zeros(shape=(self.K,)),
            'columns': np.zeros(shape=(self.L))
        }

    def action_id(self, i, j):
        return i * self.L + j

    def reset(self):
        K = self.K
        L = self.L
        self.__init__(K, L)
    

    def run(self, env, horizon=100000, animated=True):
        while self.t <= horizon:
            n_l = int(np.ceil(4 / self.delta_l**2 * np.log(horizon)))
            # candidate rows
            I_l = list(set(self.hU))
            # candidate columns
            J_l = list(set(self.hV))

            # --------- Row and column exploration --------- 
            for _ in range(n_l - self.n_lminus1):
                j = np.random.choice(self.L)
                j = self.hV[j]
                for i in I_l:
                    action = self.action_id(i, j)
                    _, reward, _, _ =env.step(action)
                    self.CU[i, j] += reward
                    # log regret
                    if self.t % env.time_series_frequency == 0:
                        env.log_regret()
                        # rendering
                        if animated: env.render()
                    self.t += 1
                    if self.t > horizon: return

                i = np.random.choice(self.K)
                i = self.hU[i]
                for j in J_l:
                    action = self.action_id(i, j)
                    _, reward, _, _ =env.step(action)
                    self.CV[i, j] += reward
                    # log regret
                    if self.t % env.time_series_frequency == 0:
                        env.log_regret()
                        # rendering
                        if animated: env.render()
                    self.t += 1
                    if self.t > horizon: return
            
            # --------- UCBs and LCBs on the expected rewards 
            # of allremaining rows and columns --------- 
            exploration = np.sqrt(np.log(horizon) / n_l)
            for i in I_l:
                self.UCB['rows'][i] = sum(self.CU[i,:]) / n_l + exploration
                self.LCB['rows'][i] = sum(self.CU[i,:]) / n_l - exploration
            for j in J_l:
                self.UCB['columns'][j] = sum(self.CV[:,j]) / n_l + exploration
                self.LCB['columns'][j] = sum(self.CV[:,j]) / n_l - exploration
            
            # --------- Row and column elimination --------- 
            i_l = I_l[0]
            for i in I_l:
                if self.LCB['rows'][i] > self.LCB['rows'][i_l]:
                    i_l = i
            for i in range(self.K):
                if self.UCB['rows'][self.hU[i]] <= self.LCB['rows'][i_l]:
                    self.hU[i] = i_l

            j_l = J_l[0]
            for j in J_l:
                if self.LCB['columns'][j] > self.LCB['columns'][j_l]:
                    j_l = j
            for j in range(self.L):
                if self.UCB['columns'][self.hV[j]] <= self.LCB['columns'][j_l]:
                    self.hV[j] = j_l
            
            self.delta_l /= 2
            self.n_lminus1 = n_l

if __name__ == '__main__':
    '''
    The following code is to replicate the experiments in Katariya et al (2017)
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--K', type=int, default=16)
    parser.add_argument('--L', type=int, default=32)
    parser.add_argument('--horizon', type=int, default=2000000)
    parser.add_argument('--p_u', type=float, default=0.7)
    parser.add_argument('--p_v', type=float, default=0.7)
    parser.add_argument('--delta_u', type=float, default=0.2)
    parser.add_argument('--delta_v', type=float, default=0.2)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--output_file', default=None)
    args = parser.parse_args()

    logger.set_level(logger.INFO)

    time_series_frequency = args.horizon // 10

    # Set up the agent
    agent = Rank1Elim(K=args.K, L=args.L)

    # Set up the environment
    env = gym.make(
        'BipartitePairBandit-v0',
        nrows = args.K,
        ncols = args.L,
        time_series_frequency = time_series_frequency
    )
    env.seed(args.seed)
    
    row_pars = [args.p_u] * args.K
    row_pars[0] += args.delta_u
    col_pars = [args.p_v] * args.L
    col_pars[0] += args.delta_v

    env.reset(agent.name, row_dist=row_pars, col_dist=col_pars)

    # Simulation loop
    agent.run(
        env=env,
        horizon=args.horizon,
        animated=True
    )

    # Render the final state and keep the plot window open
    env.render(freeze=True, output_file=args.output_file)

    env.close()


