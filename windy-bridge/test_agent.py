import gym
import windy_bridge
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env


def random_agent(episodes=1000):
    env = gym.make("windy_bridge:windy_bridge-v0")
    env.reset()
    env.render()
    cumu_reward = 0
    for e in range(episodes):
        action = env.action_space.sample()
        state, reward, done, _ = env.step(action)
        env.render()
        cumu_reward += reward
        print(cumu_reward)
        if done:
            break


def ppo_agent(episodes=1000):
    env = make_vec_env("windy_bridge:windy_bridge-v0")
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=25000)
    obs = env.reset()
    for e in range(episodes):
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        env.render()
        if dones:
            break


def custom_agent(episodes=1000):
    env = make_vec_env("windy_bridge:windy_bridge-v0")
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=25000)
    obs = env.reset()
    for e in range(episodes):
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        env.render()
        if dones:
            break


if __name__ == "__main__":
    #random_agent()
    ppo_agent()
