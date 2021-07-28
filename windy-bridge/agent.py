import gym
import windy_bridge
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import matplotlib.pyplot as plt
import json
from windy_bridge.envs.callbacks import CustomCallback


def plot_results(filename, _x,):
    with open(filename) as file_in:
        _y = json.loads(file_in.read())
    print(_x)
    print(_y)
    plt.plot(_x, _y)
    plt.show()


# todo episodes, learning steps -> check names, check values
def ppo_agent_learn(learning_steps=400000):
    """ Agent needs 131 steps to go straight to the goal (with SPEED 5)
    learning_steps can therefore be a rough multiple of 131

    eval_steps_per_run can be slightly higher than 131 to include
    cases where the agent moves up/down while already being on the same x-coord as the goal """
    env = make_vec_env("windy_bridge:windy_bridge-v0")
    model = PPO("MlpPolicy", env, verbose=1)
    callback = CustomCallback()
    model.learn(total_timesteps=learning_steps, callback=callback)


def ppo_agent_test(model, env, test_runs=10, eval_steps_per_run=200):
    obs = env.reset()
    for i in range(test_runs):
        for e in range(eval_steps_per_run):
            action, _states = model.predict(obs)
            obs, rewards, done, info = env.step(action)
            env.render()
            if done:
                break


if __name__ == "__main__":
    ppo_agent_learn()

