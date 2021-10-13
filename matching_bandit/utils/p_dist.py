'''
Useful probability distribution generator for experiments
'''
import numpy as np

def threshold2dist(v):
    n_pairs = len(v)
    p_dist = np.zeros(shape=(2*n_pairs,))
    for k in range(n_pairs):
        p_dist[2*k] = v[k]
        p_dist[2*k+1] = v[k]
    return p_dist

def centered_equal_distance(n_pairs, Delta):
    assert Delta > 0
    assert Delta * (n_pairs-1) < 1
    low = (1 - (n_pairs-1)* Delta) / 2
    high = (1 + (n_pairs-1)* Delta) / 2
    v = np.linspace(low, high, n_pairs)
    return threshold2dist(v)

def centered_equal_distance_with_zero(n_pairs, Delta):
    assert n_pairs*Delta <= 1
    p_dist = centered_equal_distance(n_pairs-1, Delta).tolist()
    p_dist += [0.0, 0.0]
    return np.array(p_dist)
    

def origin_equal_distance(n_pairs, Delta):
    high = (n_pairs - 1) * Delta
    assert  high <= 1
    v = np.linspace(0, high, n_pairs)
    return threshold2dist(v)

def ends_fixed(n_pairs, Delta):
    assert n_pairs >= 2
    second = (n_pairs - 2)*Delta
    assert second < 1
    assert n_pairs <= 2 + np.floor((1-Delta) /Delta)
    vs = list(range(0, n_pairs-1))
    vs.append(10)
    v = np.array(vs) * 0.1
    return threshold2dist(v)

def equal_distance(n_pairs, start, Delta, end=1.0):
    v = np.zeros((n_pairs,))
    for i in range(n_pairs):
        v[i] = start + i * Delta
    assert v[-1] <= end
    return threshold2dist(v)

def linear_spaced(n_pairs):
    v = np.linspace(0, 1, num=n_pairs)
    return threshold2dist(v)

def uncentered_equal_distance(n_pairs, mu, Delta):
    distance = mu - 0.5
    p_dist = centered_equal_distance(n_pairs, Delta) + distance
    assert p_dist[-1] <= 1
    assert p_dist[0] >= 0
    return p_dist

    


