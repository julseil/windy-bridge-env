import gym
import windy_bridge
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

def write_in_file(name, data):
    with open("%s.txt" % name, "w+") as writefile:
        writefile.write(str(data))

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


def ppo_agent(episodes=1000, learning_steps=1000):
    env = make_vec_env("windy_bridge:windy_bridge-v0")
    model = PPO("MlpPolicy", env, verbose=0)
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
    _x_values = [x*25 for x in range(0, 1000)]
    _y_values_ratio = []
    _y_values_reward = []
    for value in _x_values:
        ratio, avg_reward = ppo_agent(1000, value)
        _y_values_ratio.append(ratio)
        _y_values_reward.append(avg_reward[0])
        if value % 2500 == 0:
            print(value)
    write_in_file("ratio", _y_values_ratio)
    write_in_file("reward", _y_values_reward)




    # TODO graph for agent wins?
    # TODO 2 seperate trainings? with and without knowledge? is it possible in gym env?
