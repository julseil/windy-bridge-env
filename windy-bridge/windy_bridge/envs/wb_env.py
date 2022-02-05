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


def convert_to_angle(normalized_angle):
    return -90 + (normalized_angle * 180)


def convert_discrete_to_action(number, mode="dynamic"):
    if mode == "dynamic":
        if number == 0:
            return [-90, 1]
        elif number == 1:
            return [-90, 5]
        elif number == 2:
            return [-90, 10]
        elif number == 3:
            return [-90, 15]
        elif number == 4:
            return [0, 1]
        elif number == 5:
            return [0, 5]
        elif number == 6:
            return [0, 10]
        elif number == 7:
            return [0, 15]
        elif number == 8:
            return [90, 1]
        elif number == 9:
            return [90, 5]
        elif number == 10:
            return [90, 10]
        elif number == 11:
            return [90, 15]
        else:
            print("Invalid action input")
            print(number)
            return -1
    elif mode == "min":
        if number == 0:
            return [-90, 1]
        elif number == 1:
            return [-45, 1]
        elif number == 2:
            return [0, 1]
        elif number == 3:
            return [45, 1]
        elif number == 4:
            return [90, 1]
        else:
            print("Invalid action input")
            print(number)
            return -1
    elif mode == "max":
        if number == 0:
            return [-90, 10]
        elif number == 1:
            return [-45, 10]
        elif number == 2:
            return [0, 10]
        elif number == 3:
            return [45, 10]
        elif number == 4:
            return [90, 10]
        else:
            print("Invalid action input")
            print(number)
            return -1
    else:
        print("Invalid mode")
        print(mode)
        return -1


class WindyBridgeEnv(gym.Env):
    def __init__(self, mode="N/A", actionspace="continuous"):
        super(WindyBridgeEnv, self).__init__()
        self.mode = mode
        # discrete action space A_angle = {-90, 0, 90}, A_commitment = {1,5,10,15} => 12 actions
        # discrete action space A_angle = {-90, -45, 0, 45, 90}, A_commitment = {1 or 10} => 5 actions
        if actionspace == "continuous":
            self.action_space = spaces.Box(np.array([-0.9, MIN_AS[self.mode]]), np.array([0.9, MAX_AS[self.mode]]))
        elif actionspace == "discrete":
            if mode == "dynamic":
                self.action_space = spaces.Discrete(12)
            else:
                self.action_space = spaces.Discrete(5)
        else:
            print("Invalid action space parameter")
            self.action_space = -1
        self.agent = Agent(0, 0)
        self.bridge = VirtualBridge(BRIDGE_WIDTH)
        self.observation_space = spaces.Box(low=0, high=255,
                                            shape=(2,), dtype=np.float32)
        self.step_count = 0
        self.max_steps = 2048*16 # maximum number of steps per learning epoch
        self.noise_distribution = OrnsteinUhlenbeckActionNoise(theta=0.01, mu=0.1, sigma=0.1)
        self.wind_distribution_values = []
        self.current_wind_value = 0
        self.done = False

    def env_step(self, angle, commitment):
        if commitment > 0:
            self.current_wind_value = self.noise_distribution.__call__() * 4
            # todo clip wind value with step size
            if self.current_wind_value < -5:
                self.current_wind_value = -5.0
            elif self.current_wind_value > 5:
                self.current_wind_value = 5.0
            if angle < 0:
                self.agent.y += SPEED * math.sin(math.radians(angle)) + self.current_wind_value
            else:
                self.agent.y += SPEED * math.sin(math.radians(angle)) + self.current_wind_value
            self.agent.x += SPEED * math.cos(math.radians(angle))

            # logging
            self.step_count += 1
            self.wind_distribution_values.append(self.current_wind_value)

            if self.step_count >= self.max_steps-1:
                self.done = True
                # no reward for reaching the "goal"
                # self.agent.reward += 100.0
            elif not self.bridge.on_bridge(self.agent.y):
                self.done = True
                self.agent.reward -= 100.0 # /30
            else:
                self.env_step(angle, commitment - 1)

    def step(self, action):
        action = convert_discrete_to_action(action, self.mode)
        self.done = False
        self.agent.reward = -0.1
        angle = action[0]
        commitment = action[1]
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
        self.noise_distribution = OrnsteinUhlenbeckActionNoise(mu=0.4, sigma=0.4)
        self.agent.x = 0
        self.agent.old_x = 0
        self.agent.y = 0
        self.agent.pos = (self.agent.x, self.agent.y)
        self.step_count = 0
        return initial_state

    def _get_game_state(self):
        state = [self.agent.x, self.agent.y]
        return state

    def get_mode(self):
        return self.mode
