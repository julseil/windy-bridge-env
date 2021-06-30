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
    model.learn(total_timesteps=5000)
    obs = env.reset()

    wins = 0
    losses = 0
    for i in range(300):
        for e in range(episodes):
            action, _states = model.predict(obs)
            obs, rewards, done, info = env.step(action)
            #env.render()
            if done:
                if 9.8 < rewards < 10.1:
                    wins += 1
                else:
                    losses += 1
                break
    print("wins: %s" % wins)
    print("losses: %s" % losses)
    print(wins/(wins+losses))


if __name__ == "__main__":
    #random_agent()
    ppo_agent()
    # TODO graph for agent wins?
    # TODO 2 seperate trainings? with and without knowledge? is it possible in gym env?
