import gym
import matching_bandit
from matching_bandit.utils.simulation import Experiment

def matching_selection_sim(agents, p_dist, horizon, reps, save=False):
    n_items = len(p_dist)
    env = gym.make(
        'MatchingSelectionBandit-v0',
        n_pairs = n_items // 2,
        time_series_frequency = horizon // 10
    )
    env.reset('simulation', item_dist=p_dist)
    simulation = Experiment('simulation', num_reps=reps, horizon=horizon)
    simulation.run(env, agents)
    simulation.print_lastround()
    simulation.plot(save_fig=save)
    return simulation.statistics()