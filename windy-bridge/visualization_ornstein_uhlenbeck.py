# https://www.pik-potsdam.de/members/franke/lecture-sose-2016/introduction-to-python.pdf
import matplotlib.pyplot as plt
import numpy as np


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
x, y = ornstein_uhlenbeck(0, 2500, 25000, 0.2, 0, 0.3)
print(y)
plot_distribution(x, y)
