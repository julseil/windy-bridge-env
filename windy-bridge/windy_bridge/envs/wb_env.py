
import sys
import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
import pygame
import random
import time

from .ornstein_uhlenbeck import OrnsteinUhlenbeckActionNoise

# https://www.pik-potsdam.de/members/franke/lecture-sose-2016/introduction-to-python.pdf
def ornstein_uhlenbeck(t_0, t_end, length, theta, mu, sigma):
    t = np.linspace(t_0, t_end, length) # define time axis
    dt = np.mean(np.diff(t))
    y = np.zeros(length)
    y0 = np.random.normal(loc=0.0,scale=1.0) # initial condition
    drift = lambda y,t: theta*(mu-y) # define drift term, google to learn about lambda
    diffusion = lambda y,t: sigma # define diffusion term
    noise = np.random.normal(loc=0.0,scale=1.0,size=length)*np.sqrt(dt) #define noise process
    # solve SDE
    for i in range(1,length):
        y[i] = y[i-1] + drift(y[i-1],i*dt)*dt + diffusion(y[i-1],i*dt)*noise[i]
    return t, y


LENGTH = 750
WIDTH = 250
SCREEN = pygame.display.set_mode((LENGTH, WIDTH))
SPRITE_IMAGE = pygame.image.load("windy_bridge/envs/sprite.png")
SPRITE_WIDTH = 50
GOAL_IMAGE = pygame.image.load("windy_bridge/envs/flag.png")
GOAL_WIDTH = 50

BRIDGE_COLOR = "#571818"
BRIDGE_WIDTH = WIDTH/2
BRIDGE_LENGTH = LENGTH

SPEED = 5


class Bridge:
    def __init__(self):
        self.x = 0
        self.y = 0
        self.color = BRIDGE_COLOR
        self.org_width = BRIDGE_WIDTH
        self.rect_bridge = self.draw_bridge()

    def draw_bridge(self):
        self.rect_bridge = pygame.draw.rect(SCREEN, self.color, (0, self.org_width-BRIDGE_WIDTH/2, BRIDGE_LENGTH, BRIDGE_WIDTH))
        return pygame.draw.rect(SCREEN, self.color, (0, self.org_width-BRIDGE_WIDTH/2, BRIDGE_LENGTH, BRIDGE_WIDTH))


class Agent:
    def __init__(self, x=0, y=0):
        self.x = x
        self.y = y
        self.pos = (self.x, self.y)
        self.rect_agent = SPRITE_IMAGE.get_rect(topleft=(self.x, self.y))

    def draw_agent(self):
        SCREEN.blit(SPRITE_IMAGE, (self.x, self.y))


class Goal:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.pos = (self.x, self.y)
        self.rect_goal = GOAL_IMAGE.get_rect(topleft=(self.x, self.y))

    def draw_goal(self):
        SCREEN.blit(GOAL_IMAGE, (self.x, self.y))


class WindyBridgeEnv(gym.Env):
    def __init__(self):
        super(WindyBridgeEnv, self).__init__()
        self.render_delay = 0.2
        self.agent = Agent(0, WIDTH/2-SPRITE_WIDTH/2)
        self.goal = Goal(LENGTH-50, WIDTH/2-GOAL_WIDTH/2)
        self.bridge = Bridge()
        self.action_space = spaces.Discrete(8)
        # TODO richtige Werte fuer observation space? Do not hardcode size of actionspace
        self.observation_space = spaces.Box(low=0, high=255,
                                            shape=(3,), dtype=np.uint8)
        self.step_count = 0
        self.noise_distribution = OrnsteinUhlenbeckActionNoise(mu=0.2, sigma=0.2)

    def env_step(self, action, counter):
        if counter > 0:
            wind_value = self.noise_distribution.__call__()
            if action == 2:
                self.agent.x += SPEED
                self.agent.y = self.agent.y + wind_value
            if action == 3:
                self.agent.x += SPEED
                self.agent.y = self.agent.y + wind_value - SPEED
            if action == 4:
                self.agent.x += SPEED
                self.agent.y = self.agent.y + wind_value + SPEED
            if action == 5:
                self.agent.x += SPEED
                self.agent.y = self.agent.y + wind_value
            if action == 6:
                self.agent.x += SPEED
                self.agent.y = self.agent.y + wind_value - SPEED
            if action == 7:
                self.agent.x += SPEED
                self.agent.y = self.agent.y + wind_value + SPEED
            self.step_count += 1
            #self.render(delay=False)
            self.env_step(action, counter - 1)

    def step(self, action):
        reward = -0.1
        done = False
        info = {}
        # 0 = left, 1 = right,
        # 2 = 5x no direction, 3 = 5x left, 4 = 5x right,
        # 5 = 10x no direction, 6 = 10x left, 7 = 10x right
        if action == 0:
            self.agent.y = self.agent.y - SPEED + self.noise_distribution.__call__()
        if action == 1:
            self.agent.y = self.agent.y + SPEED + self.noise_distribution.__call__()
        if action == 2:
            self.env_step(action, 5)
        if action == 3:
            self.env_step(action, 5)
        if action == 4:
            self.env_step(action, 5)
        if action == 5:
            self.env_step(action, 10)
        if action == 6:
            self.env_step(action, 10)
        if action == 7:
            self.env_step(action, 10)



        self.agent.x += SPEED
        if not self._detect_fall(self.agent.x, self.agent.y):
            self.agent.pos = (self.agent.x, self.agent.y)
            self.agent.rect_agent = SPRITE_IMAGE.get_rect(topleft=(self.agent.x, self.agent.y))
            reward, done = self._check_collision(reward)
        else:
            #print("Out of bounds")
            done = True
            reward -= 10
        self.step_count += 1
        return self._get_game_state(), reward, done, info

    def reset(self):
        initial_state = []
        initial_state.append(0)
        initial_state.append(WIDTH/2-SPRITE_WIDTH/2)
        initial_state.append(0)
        self.agent.x = 0
        self.agent.y = WIDTH/2-SPRITE_WIDTH/2
        self.agent.pos = (self.agent.x, self.agent.y)
        return initial_state

    def render(self, close=False, delay=True):
        SCREEN.fill("white")
        if delay:
            time.sleep(self.render_delay)
        self.bridge.draw_bridge()
        self.agent.draw_agent()
        self.goal.draw_goal()
        pygame.display.update()

    def _detect_fall(self, x, y):
        if x > LENGTH or y > WIDTH:
            return True

    def _check_collision(self, reward):
        if self.agent.rect_agent.colliderect(self.goal.rect_goal):
            #print("Win")
            return reward+10, True
        if not self.agent.rect_agent.colliderect(self.bridge.rect_bridge):
            #print("Fell off")
            return reward-10, True
        else:
            return reward, False

    def _get_game_state(self):
        # TODO was genau ist der state? welche informationen beinhaltet state?
        # auch in funktion reset aenderen
        state = []
        state.append(self.agent.x)
        state.append(self.agent.y)
        # TODO decide wether to use previous wind value or current wind value in state
        #state.append(self.noise_distribution.x_prev)
        state.append(self.noise_distribution.x)
        return state
