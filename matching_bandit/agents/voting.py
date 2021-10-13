import argparse

import numpy as np
from collections import Counter

import gym
from gym import logger

import matching_bandit

class Voting(object):
    """
    The implementation of a voting ucb algorithm
    """
    def __init__(self, K, L):
        self.name = "Voting agent"
        self.K = K
        self.L = L
        self.row_voters = []
        self.col_voters = []

        for _ in range(K):
            self.row_voters.append(UCB(L))
        for _ in range(L):
            self.col_voters.append(UCB(K))
    
    def action_id(self, i, j):
        return i * self.L + j

    def reset(self):
        K = self.K
        L = self.L
        self.__init__(K, L)


    def run(self, env, horizon=100000, animated=True):
        for t in range(horizon):
            if t < self.K * self.L:
                action = t
                row, col = env.arm_ids(action)
            else:
                # find voted column
                candidate_cols = []
                for row_voter in self.row_voters:
                    vcol = row_voter.vote()
                    candidate_cols.append(vcol)
                col = decision(candidate_cols)
                # find voted rows
                candidate_rows = []
                for col_voter in self.col_voters:
                    vrow = col_voter.vote()
                    candidate_rows.append(vrow)
                row = decision(candidate_rows)
                action = self.action_id(row, col)
            
            _, reward, _, _ =env.step(action)
            row_voter = self.row_voters[row]
            row_voter.update(col, reward)
            col_voter = self.col_voters[col]
            col_voter.update(row, reward)

            # log regret
            if (t+1) % env.time_series_frequency == 0:
                env.log_regret()
                # rendering
                if animated: env.render()


def decision(candidates):
    # majority vote
    a = Counter(candidates)
    votee, _ = a.most_common(1)[0]
    return votee


class UCB(object):
    """
    The implementation of a UCB algorithm
    """
    def __init__(self, K):
        self.trials = np.zeros(K)
        self.rewards = np.zeros(K)

    def update(self, idx, reward):
        self.trials[idx] += 1
        self.rewards[idx] += reward

    def vote(self):
        t = sum(self.trials) + 1 # current round
        exploitation = self.rewards / self.trials
        exploration = np.sqrt(2 * np.log(t) / self.trials)
        ucb = exploitation + exploration
        return np.argmax(ucb)




if __name__ == '__main__':
    '''
    The following code is to replicate the experiments in our paper
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--K', type=int, default=32)
    parser.add_argument('--L', type=int, default=32)
    parser.add_argument('--horizon', type=int, default=2000)
    parser.add_argument('--p_u', type=float, default=0.9)
    parser.add_argument('--p_v', type=float, default=0.9)
    parser.add_argument('--delta_u', type=float, default=0.2)
    parser.add_argument('--delta_v', type=float, default=0.2)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--output_file', default=None)
    args = parser.parse_args()

    logger.set_level(logger.INFO)

    time_series_frequency = args.horizon // 10

    # Set up the agent
    agent = Voting(K=args.K, L=args.L)

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

    