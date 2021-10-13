import numpy as np

import gym
from gym import logger

import matching_bandit

import argparse

class MDC(object):

    def __init__(self, n_pairs=3):
        self.name = 'MDC'
        self.n_pairs = n_pairs
        self.n_items = n_pairs*2
        self.X = np.zeros(shape=(self.n_items, self.n_items))
        self.C = np.zeros(shape=(self.n_items, self.n_items))
        self.X_tilda = np.zeros(shape=(self.n_items, self.n_items))
        self.C_tilda = np.zeros(shape=(self.n_items, self.n_items))
        self.tournament = Tournament()
        self.tournament.head = Node(list(range(self.n_items)))

    def reset(self):
        n_pairs = self.n_pairs
        self.__init__(n_pairs=n_pairs)

    def run(self, env, horizon=10000, animated=True):
        logger.info('Algorithm: '+self.name)
        for t in range(horizon):
            action = self.tournament.sample_matching()
            # observe rewards
            obs, _, _, _ = env.step(action)
            feedback = obs['feedback']
            for k in range(self.n_pairs):
                i = action[2*k]
                j = action[2*k+1]
                self.X_tilda[i,j] += feedback[k]
                self.X_tilda[j,i] += feedback[k]     
                self.C_tilda[i,j] += 1
                self.C_tilda[j,i] += 1
            
            tmp = self.tournament.head
            while tmp is not None:
                s = tmp.cluster
                for i in s:
                    s_ = s.copy()
                    s_.remove(i)
                    counter = sum(self.C_tilda[i, s_])
                    if counter == len(s) - 1:
                        self.X[s,:] += self.X_tilda[s,:]
                        self.C[s,:] += self.C_tilda[s,:]
                        self.X_tilda[s,:] = 0
                        self.C_tilda[s,:] = 0
                        break
                tmp = tmp.next

            # make sure at least every item has been paired with all other pairs.
            if t >= self.n_items - 1:
                UCB, LCB = self.tournament.confidence_intervals(self.X, self.C, horizon)
                self.tournament.split(UCB, LCB)
            
            # log regret
            if (t+1) % env.time_series_frequency == 0:
                env.log_regret()
                # rendering
                if animated: env.render()
            


class Node:
    def __init__(self, cluster):
        self.cluster = cluster
        n_items = len(cluster)
        n_pairs = n_items // 2
        self.group = np.reshape(cluster, (2, n_pairs)).tolist()
        self.next = None

class LinkedList:
    def __init__(self):
        self.head = None

class Tournament(LinkedList):

    def sample_matching(self):
        matching = []
        tmp = self.head
        while tmp is not None:
            group = tmp.group
            # extract the matching
            matching += group2matching(group)
            # update to the next schedule
            tmp.group = round_robin_next(group)
            tmp = tmp.next
        return matching

    def confidence_intervals(self, X, C, T):
        n_items = X.shape[0]
        UCB = np.zeros(shape=(n_items,))
        LCB = np.zeros(shape=(n_items,))

        for i in range(n_items):
            lower_items = self.get_lower_items(i)
            k_l = sum(C[i,lower_items])
            d = sum(X[i,lower_items])
            LCB[i] = d/k_l - np.sqrt(np.log(T)/k_l)
            UCB[i] = d/k_l + np.sqrt(np.log(T)/k_l)

        return UCB, LCB
    
    def get_lower_items(self, idx):
        lower_items = []
        tmp = self.head
        while tmp is not None:
            cluster = tmp.cluster
            lower_items += cluster
            if idx in cluster:
                return lower_items
            tmp = tmp.next
    
    def split(self, UCB, LCB):
        tmp = self.head

        if tmp is not None:
            # if the headnode needs to be split
            TorF, high, low = is_split(tmp.cluster, UCB, LCB)
            if TorF == True:
                high_node = Node(high)
                low_node = Node(low)
                high_node.next = tmp.next
                low_node.next = high_node
                self.head = low_node

                removed = tmp
                tmp = tmp.next
                removed.next = None
                prev = high_node
            else:
                prev = tmp
                tmp = tmp.next
            
            # traverse the nodes left
            while tmp is not None:
                TorF, high, low = is_split(tmp.cluster, UCB, LCB)
                if TorF == True:
                    high_node = Node(high)
                    low_node = Node(low)
                    high_node.next = tmp.next
                    low_node.next = high_node
                    prev.next = low_node
                    
                    prev = high_node
                    tmp = tmp.next
                else:
                    prev = tmp
                    tmp = tmp.next
    
    def __str__(self):
        info = ''
        tmp = self.head
        while tmp is not None:
            info += tmp.cluster.__str__()
            tmp = tmp.next
        return info
            

def round_robin_next(group):
    '''
    Implementation of the round-robin tournament transition:
    Circle scheduling
    '''
    # rotate entries in group clockwise 
    # with the group[0,0] fixed

    group[1] = np.flip(group[1])
    group_flatten = list(np.concatenate(group).flat)
    n_items = len(group_flatten)
    if n_items > 2:
        new_group_flatten = [0]*n_items
        new_group_flatten[1:] = group_flatten[:-1]
        new_group_flatten[0] = group_flatten[-1]
        n_pairs = n_items // 2
        new_group = np.reshape(new_group_flatten, (2,n_pairs))
        new_group[1] = np.flip(new_group[1])
        tmp = new_group[0,0]
        new_group[0,0] = new_group[0,1]
        new_group[0,1] = tmp
        new_group = new_group.tolist()
    else:
        new_group = group
    
    return new_group


def is_split(cluster, UCB, LCB):
    l = len(cluster)
    if l > 2:        
        # sort items in cluster by UCB
        cluster = sorted(cluster, key=lambda i: UCB[i], reverse=True)
        for i in range(2, l, 2):
            item_i = cluster[i]
            item_im1 = cluster[i-1]
            if UCB[item_i] < LCB[item_im1]:
                # larger items put lower in the chain
                lower = cluster[:i] 
                higher = cluster[i:]
                return True, higher, lower
    
    return False, None, None


def matching2group(matching):
    n_items = len(matching)
    n_pairs = n_items // 2
    group = np.reshape([0]*n_items, (2,n_pairs))
    for k in range(n_pairs):
        group[0,k] = matching[2*k]
        group[1,k] = matching[2*k+1]
    return group.tolist()
    
def group2matching(group):
    n_pairs = len(group[0])
    n_items = n_pairs * 2
    matching = [0]*n_items
    for k in range(n_pairs):
        matching[2*k] = group[0][k]
        matching[2*k+1] = group[1][k]
    return matching



if __name__ == '__main__':
    '''
    The following code is to replicate the origin_equal_distance problem instance
    '''
    from matching_bandit.utils.p_dist import origin_equal_distance
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_pairs', type=int, default=11)
    parser.add_argument('--Delta', type=float, default=0.1)
    parser.add_argument('--horizon', type=int, default=200000)
    args = parser.parse_args()

    logger.set_level(logger.INFO)

    # Set up the agent
    agent = MDC(n_pairs=args.n_pairs)

    # Set up the environment
    env = gym.make(
        'MatchingSelectionBandit-v0',
        n_pairs = args.n_pairs,
        time_series_frequency = args.horizon // 10
    )
        
    p = origin_equal_distance(args.n_pairs, args.Delta)
    
    env.reset(agent.name, item_dist=p)

    # Simulation loop
    agent.run(
        env=env,
        horizon=args.horizon,
        animated=True
    )

    # Render the final state and keep the plot window open
    env.render(freeze=True)

    env.close()


    