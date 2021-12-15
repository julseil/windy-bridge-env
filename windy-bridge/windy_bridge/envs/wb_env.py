import math
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import pygame
import time

from .ornstein_uhlenbeck import OrnsteinUhlenbeckActionNoise

LENGTH = 750
WIDTH = 320
#SCREEN = pygame.display.set_mode((LENGTH, WIDTH))
#SPRITE_IMAGE = pygame.image.load("windy_bridge/envs/sprite.png")
SPRITE_WIDTH = 50
BRIDGE_COLOR = "#571818"
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
        #self.rect_agent = SPRITE_IMAGE.get_rect(topleft=(self.x, self.y))

    #def draw_agent(self):
        #SCREEN.blit(SPRITE_IMAGE, (self.x, self.y))


class WindyBridgeEnv(gym.Env):
    def __init__(self, mode="N/A"):
        super(WindyBridgeEnv, self).__init__()
        self.mode = mode
        # todo winkel zurück auf -0.9; 0.9 auch unten
        self.action_space = spaces.Box(np.array([-0.9, MIN_AS[self.mode]]), np.array([0.9, MAX_AS[self.mode]]))
        self.render_delay = 0.2
        self.agent = Agent(0, 0)
        self.bridge = VirtualBridge(BRIDGE_WIDTH)
        # TODO richtige Werte fuer observation space? Do not hardcode size of actionspace
        self.observation_space = spaces.Box(low=0, high=255,
                                            shape=(2,), dtype=np.float32)
        self.step_count = 0
        self.max_steps = 2000
        self.noise_distribution = OrnsteinUhlenbeckActionNoise(theta=0.01, mu=0.1, sigma=0.1, seed=np.random.randint(1000))
        self.wind_distribution_values = []
        self.done = False

    def env_step(self, angle, commitment):
        if commitment > 0:
            # todo * what number to get more varied results?
            wind_value = self.noise_distribution.__call__() * 4 # todo * 3
            if angle < 0:
                self.agent.y += SPEED * math.sin(math.radians(angle)) + wind_value
            else:
                self.agent.y += SPEED * math.sin(math.radians(angle)) + wind_value
            self.agent.x += SPEED * math.cos(math.radians(angle))
            self.step_count += 1
            #self.render()
            self.wind_distribution_values.append(wind_value)
            if self.step_count >= self.max_steps-1:
                self.done = True
                #print(f"Maximum number of steps ({self.max_steps}) was reached")
                self.agent.reward += 10.0
            elif not self.bridge.on_bridge(self.agent.y):
                self.done = True
                self.agent.reward -= 10.0
            else:
                self.env_step(angle, commitment - 1)

    #def mode(self, mode):
    #    self.mode = mode
    #    # todo winkel zurück auf -0.9; 0.9 auch oben
    #    self.action_space = spaces.Box(np.array([-0, MIN_AS[self.mode]]), np.array([0, MAX_AS[self.mode]]))
    #    return mode

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        self.done = False
        self.agent.reward = -0.5
        angle = action[0] * 100
        # todo commitment action space [1:max], no floating point numbers
        commitment = int(action[1]*10)
        self.env_step(angle, commitment)
        self.agent.reward += (self.agent.x - self.agent.old_x)/5
        distance = self.agent.x - self.agent.old_x
        self.agent.old_x = self.agent.x
        info = {"wind_values": self.wind_distribution_values, "distance": distance}
        self.wind_distribution_values = []
        self.agent.pos = (self.agent.x, self.agent.y)
        #self.agent.rect_agent = SPRITE_IMAGE.get_rect(topleft=(self.agent.x, self.agent.y))
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

    def render(self, close=False, delay=True):
        #SCREEN.fill("black")
        if delay:
            time.sleep(self.render_delay)
        self.bridge.draw_bridge()
        self.agent.draw_agent()
        pygame.display.update()


    def _get_game_state(self):
        # TODO was genau ist der state? welche informationen beinhaltet state?
        # auch in funktion reset aenderen
        state = [self.agent.x, self.agent.y]
        # TODO decide whether to use previous wind value or current wind value in state
        #state.append(self.noise_distribution.x_prev)
        #state.append(self.noise_distribution.x)
        return state
