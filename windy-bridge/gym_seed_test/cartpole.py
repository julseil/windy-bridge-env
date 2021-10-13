import gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

SEED = 1100

np.random.seed(SEED)

env = make_vec_env('CartPole-v0')
#env.seed(SEED)
#env.action_space.seed(SEED)
#env.observation_space.seed(SEED)
print(env.seed(SEED))
print(env.action_space.seed(SEED))
print(env.observation_space.seed(SEED))
env.reset()


for _ in range(500):
    env.render()
    a = env.action_space.sample()
    env.step(a) # take a random action
    #print(a)

print("-------")

env.seed(SEED)
env.action_space.seed(SEED)
env.observation_space.seed(SEED)
print(env.seed(SEED))
print(env.action_space.seed(SEED))
print(env.observation_space.seed(SEED))
env.reset()
for _ in range(5):
    #env.render()
    a = env.action_space.sample()
    env.step(a) # take a random action
    print(a)
env.close()
