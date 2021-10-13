import numpy as np
from itertools import permutations

import gym
from gym import logger

import matching_bandit

import argparse


class ESCB(object):
    def __init__(self, n_pairs=3, case='c'):
        self.name = 'ESCB'
        self.type = case
        self.n_pairs = n_pairs
        self.n_items = n_pairs*2
        all_pairs = self.n_items * (self.n_items - 1) // 2
        self.trials = np.zeros(shape=(all_pairs,))
        self.rewards = np.zeros(shape=(all_pairs,))
        self.n = 1
        self.arm_set = self.generate_matchings()

    def reset(self):
        all_pairs = self.n_items * (self.n_items - 1) // 2
        self.trials = np.zeros(shape=(all_pairs,))
        self.rewards = np.zeros(shape=(all_pairs,))
        self.n = 1

    def run(self, env, horizon=100000, animated=True):
        logger.info('Algorithm: '+self.name)
        n_arms = len(self.arm_set)
        while self.n <= horizon:
            # try all matching once to ensure t_i > 0
            if self.n <= n_arms:
                action = list(self.arm_set[self.n - 1])
            else:
                # find the highest index
                action = self.act()
            # observe rewards
            obs, _, _, _ = env.step(action)
            # update the states
            self.update(obs)
            # log regret
            if self.n % env.time_series_frequency == 0:
                env.log_regret()
                # rendering
                if animated: env.render()
            self.n += 1

    def act(self):
        fn = np.log(self.n) # the author replace f(n) with log(n) in experiments
        indexes = []
        for arm in self.arm_set:
            embedding = self.one_hot(arm, case='matching')
            idx = self.index(embedding, fn)
            indexes.append(idx)
        id_max = np.argmax(indexes)
        return list(self.arm_set[id_max])

    def update(self, obs):
        self.trials = self.one_hot(obs['trials'], case='symmetric-matrix')
        self.rewards = self.one_hot(obs['rewards'], case='symmetric-matrix')

    def index(self, M, fn):
        if self.type == 'c':
            return self.c_index(M, fn)
        elif self.type == 'b':
            return self.b_index(M, fn)
        else:
            raise NotImplementedError

    def b_index(self, M, fn):
        return

    def c_index(self, M, fn):
        thetahat = self.rewards / self.trials
        exploit = np.inner(M, thetahat)
        exploration = np.sqrt(fn * sum(M/self.trials) /2)
        return exploit + exploration

    def one_hot(self, item, case):
        d = len(self.trials)
        embedding = np.zeros(shape=(d,))
        if case == 'matching':
            for k in range(self.n_pairs):
                i = item[2*k]
                j = item[2*k+1]
                id = self.pair_id(i, j)
                embedding[id] = 1

        elif case == 'symmetric-matrix':
            for i in range(self.n_items-1):
                for j in range(i+1, self.n_items):
                    id = self.pair_id(i, j)
                    embedding[id] = item[i,j]
        else:
            raise NotImplementedError

        return np.array(embedding)
    
    def pair_id(self, i, j):
        assert i < j
        assert j < self.n_items
        counter = 0
        i_counter = 0
        while i_counter < i:
            counter += self.n_items - 1 - i_counter
            i_counter += 1
        j_counter = i + 1
        while j_counter < j:
            counter += 1
            j_counter += 1
        return counter

    def generate_matchings(self):
        perms = permutations(range(self.n_items))
        saved = []
        matchings = []
        for tup in perms:
            l = list(tup)
            idxes = []
            for k in range(self.n_pairs):
                if l[2*k] > l[2*k+1]:
                    tmp = l[2*k] 
                    l[2*k] = l[2*k+1]
                    l[2*k+1] = tmp
                i = l[2*k]
                j = l[2*k+1]
                idxes.append(self.pair_id(i, j))
            ivt = set(idxes)
            if ivt not in saved:
                saved.append(ivt)
                matchings.append(tuple(l))
        return matchings


if __name__ == '__main__':
    '''
    The following code is to replicate the experiments in Katariya et al (2017)
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_pairs', type=int, default=3)
    parser.add_argument('--horizon', type=int, default=10000)
    parser.add_argument('--type', type=str, default='c')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--output_file', default=None)
    args = parser.parse_args()

    logger.set_level(logger.INFO)

    time_series_frequency = args.horizon // 10

    # Set up the agent
    agent = ESCB(n_pairs=args.n_pairs, case=args.type)

    # Set up the environment
    env = gym.make(
        'MatchingSelectionBandit-v0',
        n_pairs = args.n_pairs,
        time_series_frequency = time_series_frequency
    )
    env.seed(args.seed)

    # p = [0.44115209, 0.82543791, 0.28106102, 0.18624691]
    p = None
    env.reset(agent.name, item_dist=p)

    # Simulation loop
    agent.run(
        env=env,
        horizon=args.horizon,
        animated=True
    )

    # Render the final state and keep the plot window open
    env.render(freeze=True, output_file=args.output_file)

    env.close()

