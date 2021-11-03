import gym
import windy_bridge
from stable_baselines3 import PPO
from stable_baselines3 import A2C
from stable_baselines3.common.env_util import make_vec_env
import matplotlib.pyplot as plt
import json
from windy_bridge.envs.callbacks import CustomCallback


def ppo_agent_learn(seeds, modes, learning_steps=1024000*4): # 1024000*2 737280
    """ learning steps is a multiple of 2048 (steps before update)
    eval_steps_per_run can be slightly higher than 131 to include
    cases where the agent moves up/down while already being on the same x-coord as the goal """
    i = 1
    for seed in seeds:
        print("----------------------")
        print(f">>> seed: {seed}, {i}/{len(seeds)}")
        for mode in modes:
            print(f">>>>> mode: {mode}")
            env = make_vec_env("windy_bridge:windy_bridge-v0", env_kwargs={"mode": mode})
            env.seed(seed)
            env.action_space.seed(seed)
            env.observation_space.seed(seed)
            # todo test A2C
            model = PPO("MlpPolicy", env, verbose=0, seed=seed)
            callback = CustomCallback(seed=seed, mode=mode)
            model.learn(total_timesteps=learning_steps, callback=callback)
        i += 1


if __name__ == "__main__":
    seeds = [33, 105, 74, 8, 21]
    seeds = [12, 99, 4, 34, 15]
    modes = ["min", "max", "dynamic"]
    ppo_agent_learn(seeds, modes)
