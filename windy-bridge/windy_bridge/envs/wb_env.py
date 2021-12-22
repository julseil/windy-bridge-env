import math
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import pygame
import time

from .ornstein_uhlenbeck import OrnsteinUhlenbeckActionNoise


# global vars
LENGTH = 750
WIDTH = 320
BRIDGE_WIDTH = WIDTH/2
BRIDGE_LENGTH = LENGTH

SPEED = 5
MAX_STEP = 10

MIN_AS = {"min": 0.1, "max": MAX_STEP/10, "dynamic": 0.1}
MAX_AS = {"min": 0.1, "max": MAX_STEP/10, "dynamic": MAX_STEP}


class VirtualBridge:
    def __init__(self, width):
        self.width = width
        self.positive_bound = 0 + width/2
        self.negative_bound = 0 - width/2

    def on_bridge(self, y):
        if self.negative_bound < y < self.positive_bound:
            return True
        else:
            return False


class Agent:
    def __init__(self, x=0, y=0):
        self.x = x
        self.y = y
        self.old_x = 0
        self.pos = (self.x, self.y)
        self.reward = 0


class WindyBridgeEnv(gym.Env):
    def __init__(self, mode="N/A"):
        super(WindyBridgeEnv, self).__init__()
        self.mode = mode
        self.action_space = spaces.Box(np.array([-0.9, MIN_AS[self.mode]]), np.array([0.9, MAX_AS[self.mode]]))
        self.agent = Agent(0, 0)
        self.bridge = VirtualBridge(BRIDGE_WIDTH)
        self.observation_space = spaces.Box(low=0, high=255,
                                            shape=(2,), dtype=np.float32)
        self.step_count = 0
        self.max_steps = 2048*16 # maximum number of steps per learning epoch
        self.noise_distribution = OrnsteinUhlenbeckActionNoise(theta=0.01, mu=0.1, sigma=0.1, seed=np.random.randint(1000))
        self.wind_distribution_values = []
        self.done = False

    def env_step(self, angle, commitment):
        if commitment > 0:
            wind_value = self.noise_distribution.__call__() * 4
            if angle < 0:
                self.agent.y += SPEED * math.sin(math.radians(angle)) + wind_value
            else:
                self.agent.y += SPEED * math.sin(math.radians(angle)) + wind_value
            self.agent.x += SPEED * math.cos(math.radians(angle))

            # logging
            self.step_count += 1
            self.wind_distribution_values.append(wind_value)

            if self.step_count >= self.max_steps-1:
                self.done = True
                self.agent.reward += 100.0
            elif not self.bridge.on_bridge(self.agent.y):
                self.done = True
                self.agent.reward -= 100.0
            else:
                self.env_step(angle, commitment - 1)

    def step(self, action):
        self.done = False
        self.agent.reward = -0.1
        angle = action[0] * 100
        commitment = int(action[1]*10)
        self.env_step(angle, commitment)
        self.agent.reward += (self.agent.x - self.agent.old_x)
        distance = self.agent.x - self.agent.old_x
        self.agent.old_x = self.agent.x
        info = {"wind_values": self.wind_distribution_values, "distance": distance}
        self.wind_distribution_values = []
        self.agent.pos = (self.agent.x, self.agent.y)
        self.step_count += 1
        return self._get_game_state(), self.agent.reward, self.done, info

    def reset(self):
        initial_state = [0, 0]
        self.noise_distribution = OrnsteinUhlenbeckActionNoise(mu=0.4, sigma=0.4, seed=np.random.randint(1000))
        self.agent.x = 0
        self.agent.old_x = 0
        self.agent.y = 0
        self.agent.pos = (self.agent.x, self.agent.y)
        self.step_count = 0
        return initial_state

    def _get_game_state(self):
        state = [self.agent.x, self.agent.y]
        return state
