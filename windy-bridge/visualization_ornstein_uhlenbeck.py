# https://www.pik-potsdam.de/members/franke/lecture-sose-2016/introduction-to-python.pdf
import matplotlib.pyplot as plt
import numpy as np


def ornstein_uhlenbeck(t_0, t_end, length, theta, mu, sigma):
    t = np.linspace(t_0, t_end, length) # define time axis
    dt = np.mean(np.diff(t))
    y = np.zeros(length)
    y0 = np.random.normal(loc=0.0, scale=1.0) # initial condition
    drift = lambda y, t: theta*(mu-y) # define drift term, google to learn about lambda
    diffusion = lambda y, t: sigma # define diffusion term
    noise = np.random.normal(loc=0.0, scale=1.0, size=length)*np.sqrt(dt) #define noise process
    print(noise)
    # solve SDE
    for i in range(1, length):
        y[i] = y[i-1] + drift(y[i-1], i*dt)*dt + diffusion(y[i-1], i*dt)*noise[i]
    _y = [i * 3 for i in y]
    return t, _y


def plot_distribution(_x, _y):
    plt.plot(_x, _y)
    plt.show()

# [0] start x value;
# [1] end x value;
# [2] number of values between [0] and [1];
# [3] mean reversion speed;
# [4] mean reversion level;
# [5] influence of randomness (0 = flat line)
# (0, total_timesteps/10, total_timesteps, 1.1, 0, 0.3)
np.random.seed(1)
x, y = ornstein_uhlenbeck(0, 2500, 2500, 0.01, 0.1, 0.1)
print(max(y))
print(min(y))
plot_distribution(x,y)

# todo get rid of length variable
# for t -> append with each next? / update linspace with each new step
# for dt -> ?
# for y -> only store current y value (like fib a,b = b, a+b)
# for noise -> ?
# for increment i -> incrementor from generator (e.g. infinite sequence num)
class Ornuhl:
    def __init__(self, t_0, theta, mu, sigma):
        self.t_0 = t_0
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.t_end = 0
        self.t = []
        self.y = 0
        self.dt = np.mean(np.diff(self.t))
        self.drift = lambda y,t: self.theta*(self.mu-self.y)
        self.diffusion = lambda y,t: self.sigma
        self.noise = np.random.normal(loc=0.0,scale=1.0,size=self.length)*np.sqrt(self.dt)

    def __next__(self):
        self.y = self.y + self.drift(self.y, "?" * self.dt) * self.dt + self.diffusion(self.y, "?" * self.dt) * self.noise["?"]

    def next(self):
        return self.__next__()

    def update_t(self):
        self.t = np.linspace(self.t_0, self.t_end, self.t_end / self.dt)


def infinite_sequence():
    num = 1
    while True:
        yield num
        num += 1

