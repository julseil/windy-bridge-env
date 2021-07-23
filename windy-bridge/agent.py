import gym
import windy_bridge
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import matplotlib.pyplot as plt
import json


def write_in_file(name, data):
    with open("%s.txt" % name, "w+") as writefile:
        writefile.write(str(data))


def plot_results(filename, _x,):
    with open(filename) as file_in:
        _y = json.loads(file_in.read())
    print(_x)
    print(_y)
    plt.plot(_x, _y)
    plt.show()


# todo episodes, learning steps -> check names, check values
# todo what is number of steps needed for straight line
def ppo_agent(episodes=1000, learning_steps=1000):
    env = make_vec_env("windy_bridge:windy_bridge-v0")
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=learning_steps)
    obs = env.reset()

    test_runs = 500
    wins = 0
    losses = 0
    avg_reward = 0
    for i in range(test_runs):
        for e in range(episodes):
            action, _states = model.predict(obs)
            obs, rewards, done, info = env.step(action)
            env.render()
            if done:
                if 9.8 < rewards < 10.1:
                    wins += 1
                else:
                    losses += 1
                avg_reward += rewards
                break
    avg_reward = avg_reward/test_runs
    ratio = wins/(wins+losses)
    return ratio, avg_reward


if __name__ == "__main__":
    ratio, avg_reward = ppo_agent()
