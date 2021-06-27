from gym.envs.registration import register

register(
    id='windy_bridge-v0',
    entry_point='windy_bridge.envs:WindyBridgeEnv',
)
