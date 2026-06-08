"""The Pi agent harness (tools + sandboxed runner + reward + SkyRL env adapter).

Same harness is used for eval and RL rollouts (fair comparison). Trajectories are
captured Polar-style (token-faithful) so loss masks/advantages align in RL.
"""
