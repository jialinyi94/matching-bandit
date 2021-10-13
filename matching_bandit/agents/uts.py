import argparse

import numpy as np

import gym
from gym import logger

import matching_bandit


class UTS(object):
    """
    Implementation of Unimodal Thompson sampling in Trinh et al (2020)
    """
    def __init__(self, K, L, gamma):
        self.name = 'UTS'
        self.K = K
        self.L = L
        self.gamma = gamma
        self.trials = np.zeros((K, L))
        self.leader_counts = np.zeros((K, L))
        self.rewards = np.zeros((K, L))
    
    def action_id(self, i, j):
        return i * self.L + j

    def reset(self):
        K = self.K
        L = self.L
        self.trials = np.zeros((K, L))
        self.leader_counts = np.zeros((K, L))
        self.rewards = np.zeros((K, L))
    
    def run(self, env, horizon=100000, animated=True):
        for t in range(horizon):
            if t < self.K * self.L:
                action = t
                row, col = env.arm_ids(action)
            else:
                mu_hat = self.rewards / self.trials
                leader_action = np.argmax(mu_hat)
                row_leader, col_leader = env.arm_ids(leader_action)
                self.leader_counts[row_leader, col_leader] += 1
                if self.leader_counts[row_leader, col_leader] % self.gamma == 0:
                    action = leader_action
                else:
                    row = row_leader
                    col = 0
                    theta_max = 0
                    # search along the leader row
                    for j in range(self.L):
                        a = self.rewards[row_leader, j] + 1
                        b = self.trials[row_leader, j] - self.rewards[row_leader, j] + 1
                        theta = np.random.beta(a, b)
                        if theta > theta_max:
                            theta_max = theta
                            col = j
                    # search along the leader column, skip the duplicate entry
                    for i in range(self.K):
                        if i != row_leader:
                            a = self.rewards[i, col_leader] + 1
                            b = self.trials[i, col_leader] - self.rewards[i, col_leader] + 1
                            theta = np.random.beta(a, b)
                            if theta > theta_max:
                                row = i
                                col = col_leader
                    action = self.action_id(row, col)
            
            _, reward, _, _ =env.step(action)
            self.trials[row, col] += 1
            self.rewards[row, col] += reward

            # log regret
            if (t+1) % env.time_series_frequency == 0:
                env.log_regret()
                # rendering
                if animated: env.render()

                    
if __name__ == '__main__':
    '''
    The following code is to replicate the experiments in Trinh et al (2020)
    '''

    parser = argparse.ArgumentParser()
    parser.add_argument('--K', type=int, default=4)
    parser.add_argument('--L', type=int, default=4)
    parser.add_argument('--gamma', type=int, default=2)
    parser.add_argument('--horizon', type=int, default=300000)
    parser.add_argument('--p_u', type=float, default=0.25)
    parser.add_argument('--p_v', type=float, default=0.25)
    parser.add_argument('--delta_u', type=float, default=0.5)
    parser.add_argument('--delta_v', type=float, default=0.5)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--output_file', default=None)
    args = parser.parse_args()

    logger.set_level(logger.INFO)

    time_series_frequency = args.horizon // 10

    # Set up the environment
    env = gym.make(
        'BipartitePairBandit-v0',
        nrows = args.K,
        ncols = args.L,
        time_series_frequency = time_series_frequency
    )

    #env.seed(args.seed)

    # Set up the agent
    agent = UTS(K=args.K, L=args.L, gamma=args.gamma)

    row_pars = np.array([args.p_u] * args.K)
    row_pars[1] += args.delta_u
    col_pars = np.array([args.p_v] * args.L)
    col_pars[1] += args.delta_v

    env.reset(agent.name, row_dist=row_pars, col_dist=col_pars)

    agent.run(
        env=env,
        horizon=args.horizon
    )

    # Render the final state and keep the plot window open
    env.render(freeze=True, output_file=args.output_file)

    env.close()
