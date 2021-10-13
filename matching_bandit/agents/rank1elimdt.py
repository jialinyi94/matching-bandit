import argparse

import numpy as np

import gym
from gym import logger

import matching_bandit
from matching_bandit.agents.seb_elim import SEBElim 

import copy


class Rank1ElimDT(object):
    """
    Implementation of Rank1ElimDT algorithm
    """
    def __init__(self, R, C,horizon):
        self.name = 'PairElim'
        self.R = R
        self.C = C
        self.t = 1
        self.R_a = list(range(R))
        self.C_a = list(range(C))
        self.w = 0
        self.horizon = horizon

        self.SEB_col = []
        self.SEB_row = []

        for i in range(R+1):
            self.SEB_row.append(SEBElim(i, 2**(2**self.w), self.C_a,C, j_type='row'))

        for j in range(C+1):
            self.SEB_col.append(SEBElim(j, horizon, self.R_a,R, j_type='column'))


    def action_id(self, i, j):
        return i * self.C+ j

    def new_time_window(self):
        self.C_a = list(range(self.C))
        self.SEB_row = []
        self.w += 1
        print("time window",self.w, 2**(2**self.w))
        for i in range(self.R+1):
            self.SEB_row.append(SEBElim(i, 2**(2**self.w), self.C_a,self.C, j_type='row'))

    def reset(self):
        R = self.R
        C = self.C
        horizon = self.horizon
        self.__init__(R, C, horizon)

    def run(self, env, horizon, animated=True):
        while self.t <= horizon:
            #intiate sampling stage
            X = np.zeros((self.R,self.C))
            I_l = list(set(self.R_a))
            # candidate columns
            J_l = list(set(self.C_a))
            j = np.random.choice(self.C)
            j = self.C_a[j]
            for i in I_l:
                action = self.action_id(i, j)
                _, reward, _, _ =env.step(action)
                X[i, j] += reward
                # log regret
                if self.t % env.time_series_frequency == 0:
                    env.log_regret()
                    # rendering
                    if animated: env.render()
                self.t += 1
                if self.t > horizon: return

            i = np.random.choice(self.R)
            i = self.R_a[i]
            for k in J_l:
                action = self.action_id(i, k)
                _, reward, _, _ =env.step(action)
                X[i, k] = reward
                # log regret
                if self.t % env.time_series_frequency == 0:
                    env.log_regret()
                    # rendering
                    if animated: env.render()
                self.t += 1
                if self.t > horizon: return

            #Update SEB-Elim processes
            D_C= []
            E_c = -1
            D_R= []
            E_r = -1
 
            d_c, e_c= self.SEB_row[i].update(X[i,:])
            if len(d_c)>0:
               E_c = e_c
               D_C = D_C+d_c
            d_c, e_c= self.SEB_row[-1].update(X[i,:])
            if len(d_c)>0:
                E_c = e_c
                D_C = D_C+d_c


            d_r, e_r = self.SEB_col[j].update(X[:,j])
            D_R= D_R+d_r
            if len(d_r)>0:
                E_r = e_r
                D_R= D_R+d_r
            d_r, e_r = self.SEB_col[-1].update(X[:,j])
            if len(d_r)>0:
                E_r = e_r
                D_R= D_R+d_r
            #Transfer deleted rows and columns information
            for i in range(self.R+1):
                self.SEB_row[i].remove(list(set(D_C)))
            for j in range(self.C+1):
                self.SEB_col[j].remove(list(set(D_R)))

            for j in range(self.C):
                if self.C_a[j] in D_C:
                    self.C_a[j] = E_c


            for i in range(self.R):
                if self.R_a[i] in D_R:
                    self.R_a[i] = E_r

            #Change time window
            if self.t > 2**(2**self.w) : self.new_time_window()



if __name__ == '__main__':
    '''
    The following code is to replicate the experiments in our paper
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--K', type=int, default=8)
    parser.add_argument('--L', type=int, default=8)
    parser.add_argument('--horizon', type=int, default=100000)
    parser.add_argument('--p_u', type=float, default=0.9)
    parser.add_argument('--p_v', type=float, default=0.9)
    parser.add_argument('--delta_u', type=float, default=0.5)
    parser.add_argument('--delta_v', type=float, default=0.5)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--output_file', default=None)
    args = parser.parse_args()

    logger.set_level(logger.INFO)

    time_series_frequency = args.horizon // 10

    # Set up the agent
    agent = Rank1ElimDT(R=args.K, C=args.L, horizon=args.horizon)

    # Set up the environment
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