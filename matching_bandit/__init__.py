from gym.envs.registration import register

register(
    id='MonopartitePairBandit-v0',
    entry_point='matching_bandit.envs:MonopartitePairBanditEnv'
)

register(
    id='BipartitePairBandit-v0',
    entry_point='matching_bandit.envs:BipartitePairBanditEnv'
)

register(
    id='MatchingSelectionBandit-v0',
    entry_point='matching_bandit.envs:MatchingSelectionBanditEnv'
)