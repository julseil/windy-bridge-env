
import sys
import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
import pygame
import random
import time

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

SPEED = 5.5
# TODO avoid hardcoding
# TODO # (0, total_timesteps/10+1, total_timesteps+10, 1.1/0.2, 0, 0.3)
WIND_DISTRIBUTION = ornstein_uhlenbeck(0, 5000, 50000, 0.2, 0, 0.3)


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
        self.action_space = spaces.Discrete(4)
        # TODO richtige Werte fuer observation space?
        self.observation_space = spaces.Box(low=0, high=255,
                                            shape=(4,), dtype=np.uint8)
        self.step_count = 0

    def step(self, action):
        reward = -0.1
        done = False
        info = {}
        x = self.agent.x
        y = self.agent.y
        # 0 = right, 1 = left, 2 = up, 3 = down
        if action == 0:
            x = x + SPEED + SPEED
        if action == 1:
            x = x - SPEED + SPEED
        if action == 2:
            y -= SPEED
            x += SPEED
        if action == 3:
            y += SPEED
            x += SPEED
        if not self._detect_fall(x, y):
            self.agent.x = x
            # todo get wind values in sequence not random
            self.agent.y = y + WIND_DISTRIBUTION[1][self.step_count]
            self.agent.pos = (self.agent.x, self.agent.y)
            self.agent.rect_agent = SPRITE_IMAGE.get_rect(topleft=(self.agent.x, self.agent.y))
            reward, done = self._check_collision(reward)

        else:
            print("Out of bounds")
            done = True
            reward -= 10
        self.step_count += 1
        return self._get_game_state(), reward, done, info

    def reset(self):
        initial_state = []
        initial_state.append(0)
        initial_state.append(WIDTH/2-SPRITE_WIDTH/2)
        initial_state.append(LENGTH-50)
        initial_state.append(WIDTH/2-GOAL_WIDTH/2)
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
            print("Win")
            return reward+10, True
        if not self.agent.rect_agent.colliderect(self.bridge.rect_bridge):
            print("Fell off")
            return reward-10, True
        else:
            return reward, False

    def _get_game_state(self):
        # TODO was genau ist der state? welche informationen beinhaltet state?
        # auch in funktion reset aenderen
        state = []
        state.append(self.agent.x)
        state.append(self.agent.y)
        state.append(self.goal.x)
        state.append(self.goal.y)
        return state
