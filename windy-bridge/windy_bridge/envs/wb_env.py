import math
import gym
from gym import spaces
import numpy as np
import pygame
import time

from .ornstein_uhlenbeck import OrnsteinUhlenbeckActionNoise

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
MAX_STEP = 10
# action space shape
# normal
MIN_AS = 0.1
MAX_AS = MAX_STEP/10
# min baseline
MIN_AS = 0.1
MAX_AS = 0.1
## max baseline
#MIN_AS = MAX_STEP/10
#MAX_AS = MAX_STEP/10

MIN_ASS = [0.1, 0.1, MAX_STEP/10]
MAX_ASS = [MAX_STEP/10, 0.1, MAX_STEP/10]


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
        self.action_space = spaces.Box(np.array([-0.9, MIN_AS]), np.array([0.9, MAX_AS]))
        # TODO richtige Werte fuer observation space? Do not hardcode size of actionspace
        self.observation_space = spaces.Box(low=0, high=255,
                                            shape=(2,), dtype=np.int32)
        self.step_count = 0
        self.noise_distribution = OrnsteinUhlenbeckActionNoise(mu=0.2, sigma=0.2)

    def env_step(self, angle, commitment):
        if commitment > 0:
            # todo * what number to get more varied results?
            wind_value = self.noise_distribution.__call__() * 5
            wind_value = 0
            if angle < 0:
                self.agent.y += SPEED * math.sin(math.radians(angle)) + wind_value
            else:
                self.agent.y += SPEED * math.sin(math.radians(angle)) + wind_value
            self.agent.x += SPEED * math.cos(math.radians(angle))
            self.agent.rect_agent = SPRITE_IMAGE.get_rect(topleft=(self.agent.x, self.agent.y))
            self.step_count += 1
            #self.render()
            self.env_step(angle, commitment - 1)

    def step(self, action):
        reward = -0.1
        done = False
        info = {}
        commitment = int(action[1]*10)
        angle = action[0] * 100
        self.env_step(angle, commitment)

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
        #state.append(self.noise_distribution.x)
        return state
