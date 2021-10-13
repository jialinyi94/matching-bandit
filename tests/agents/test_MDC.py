import unittest
import matching_bandit
import numpy as np
from matching_bandit.agents.simple_adaptive_matching import round_robin_next
from matching_bandit.agents.simple_adaptive_matching import group2matching, matching2group
from matching_bandit.agents.simple_adaptive_matching import is_split
from matching_bandit.agents.simple_adaptive_matching import Node, Tournament

class TestFunctions(unittest.TestCase):
    def test_round_robin_next(self):
        group = [[1, 2, 3], [6, 5, 4]]
        expected = [[1, 6, 2], [5, 4, 3]]
        self.assertEqual(round_robin_next(group), expected)

    def test_matching2group(self):
        m = [1,2,4,5,9,8]
        expected = [[1,4,9], [2, 5, 8]]
        self.assertEqual(matching2group(m), expected)

    def test_group2matching(self):
        group = [[1, 2, 3], [6, 5, 4]]
        expected = [1,6,2,5,3,4]
        self.assertEqual(group2matching(group), expected)
    
    def test_is_split(self):
        cluster = [0, 1, 2, 3, 4, 5, 6, 7]
        UCB = np.array([0.9, 0.6, 0.8, 0.3, 0.72, 0.1, 0.2, 0.4])
        LCB = UCB - 0.1
        TorF, higher, lower = is_split(cluster, UCB, LCB)
        self.assertEqual(TorF, True)
        self.assertEqual(higher, [0, 2, 4, 1])
        self.assertEqual(lower, [7, 3, 6, 5])

    def test_sample_matching(self):
        root = Node(list(range(8)))
        tournament = Tournament()
        tournament.head = root
        expected = [0, 4, 1, 5, 2, 6, 3, 7]
        self.assertEqual(tournament.sample_matching(), expected)
        expected = [0, 5, 4, 6, 1, 7, 2, 3]
        self.assertEqual(tournament.sample_matching(), expected)

    def test_split(self):
        root = Node(list(range(8)))
        tournament = Tournament()
        tournament.head = root
        expected = [0, 1, 2, 3, 4, 5, 6, 7].__str__()
        self.assertEqual(tournament.__str__(), expected)
        
        UCB = np.array([0.9, 0.6, 0.8, 0.3, 0.72, 0.1, 0.2, 0.4])
        LCB = UCB - 0.1
        tournament.split(UCB, LCB)
        expected = '[7, 3, 6, 5][0, 2, 4, 1]'
        self.assertEqual(tournament.__str__(), expected)

        UCB = np.array([0.3, 0.2, 0.9, 0.2, 0.8, 0.3, 0.6, 0.7])
        LCB = UCB - 0.1
        tournament.split(UCB, LCB)
        expected = '[5, 3][7, 6][0, 1][2, 4]'
        self.assertEqual(tournament.__str__(), expected)  

        UCB = np.array([0.9, 0.3, 0.9, 0.3, 0.3, 0.9, 0.3, 0.9])
        LCB = UCB - 0.1
        tournament.split(UCB, LCB)
        expected = '[5, 3][7, 6][0, 1][2, 4]'
        self.assertEqual(tournament.__str__(), expected)

        expected = [5, 3, 7, 6, 0, 1, 2, 4]
        self.assertEqual(tournament.sample_matching(), expected)

        all_items = list(range(8))
        tournament = Tournament()
        tournament.head = Node(all_items)

        theta = np.array([0.4, 0.7, 0.9, 0.2, 0.2, 0.9, 0.7, 0.4])
        radius = [0.2, 0.12, 0.05, 0.0025, 0.0025, 0.0025]
        expected_matchings = [
            [0, 4, 1, 5, 2, 6, 3, 7],
            [0, 5, 4, 6, 1, 7, 2, 3],
            [0, 3, 7, 4, 2, 1, 5, 6],
            [3, 4, 0, 7, 1, 6, 2, 5],
            [3, 4, 0, 7, 1, 6, 2, 5],
            [3, 4, 0, 7, 1, 6, 2, 5]
        ]
        expected_tournaments = [
            '[0, 1, 2, 3, 4, 5, 6, 7]',
            '[0, 7, 3, 4][2, 5, 1, 6]',
            '[3, 4][0, 7][1, 6][2, 5]',
            '[3, 4][0, 7][1, 6][2, 5]',
            '[3, 4][0, 7][1, 6][2, 5]',
            '[3, 4][0, 7][1, 6][2, 5]'
        ]

        for i in range(4):
            m = tournament.sample_matching()
            self.assertEqual(m, expected_matchings[i])
            UCB = theta + radius[i]
            LCB = theta - radius[i]
            tournament.split(UCB, LCB)
            self.assertEqual(tournament.__str__(), expected_tournaments[i])

    def test_get_lower_items(self):
        tournament = Tournament()
        tournament.head = Node([3, 4])
        tournament.head.next = Node([0, 7])
        tournament.head.next.next = Node([1, 6])
        tournament.head.next.next.next = Node([2, 5])

        expected = [3, 4, 0, 7]
        self.assertEqual(tournament.get_lower_items(0), expected)
        


if __name__ == '__main__':
    unittest.main()