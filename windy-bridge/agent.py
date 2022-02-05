import gym
import windy_bridge
from stable_baselines3 import PPO, A2C, DQN
from stable_baselines3.common.env_util import make_vec_env
from windy_bridge.envs.callbacks import CustomCallback

# todo
# max = baseline (name)
# lineplots
# test different buffer size for DQN
# DQN plot epsilon vs. discounted/cumu reward


def ppo_agent_learn(modes, learning_steps=1024000, seeds=None): # 1024000/4
    """ learning steps is a multiple of 2048 (steps before update)
    eval_steps_per_run can be slightly higher than 131 to include
    cases where the agent moves up/down while already being on the same x-coord as the goal """
    if seeds:
        i = 1
        for seed in seeds:
            print("----------------------")
            print(f">>> seed: {seed}, {i}/{len(seeds)}")
            for mode in modes:
                print(f">>>>> (with seeds) mode: {mode}")
                env = make_vec_env("windy_bridge:windy_bridge-v0", env_kwargs={"mode": mode})
                env.seed(seed)
                env.action_space.seed(seed)
                env.observation_space.seed(seed)
                # todo test A2C
                #model = PPO("MlpPolicy", env, verbose=0, n_steps=2048*4)
                #callback = CustomCallback(seed=seed, mode=mode)
                #model.learn(total_timesteps=learning_steps, callback=callback)
            i += 1
    else:
        for mode in modes:
            print(f">>>>> (without seeds) mode: {mode}")
            print(">>>>> Using discrete action space")
            env = make_vec_env("windy_bridge:windy_bridge-v0", env_kwargs={"mode": mode, "actionsapce": "continuous"})
            model = PPO("MlpPolicy", env, verbose=0, n_steps=2048*2)
            #model = A2C("MlpPolicy", env, verbose=0, n_steps=5)
            #model = DQN("MlpPolicy", env, verbose=0, buffer_size=learning_steps/5)
            callback = CustomCallback(mode=mode)
            model.learn(total_timesteps=learning_steps, callback=callback)


if __name__ == "__main__":
    # set mode(s) ["min", "max", "dynamic"]
    modes = ["min", "max", "dynamic"]

    # run with one random seed
    #ppo_agent_learn(modes=modes, seeds=seeds)
    # run without seeds
    ppo_agent_learn(modes=modes)
