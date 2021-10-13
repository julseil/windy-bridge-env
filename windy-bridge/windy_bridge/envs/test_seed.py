from ornstein_uhlenbeck import OrnsteinUhlenbeckActionNoise
import matplotlib.pyplot as plt
import numpy as np

noise_distribution = OrnsteinUhlenbeckActionNoise(mu=0.2, sigma=0.2, theta=0.15, seed=311212)

max = -1000
min = 1000
avg = 0
it = 5000
x = []
y = []


for i in range(0,it):
    tmp = noise_distribution.__call__() * 3
    if tmp < min:
        min = tmp
    if tmp > max:
        max = tmp
    avg += tmp
    x.append(i)
    y.append(tmp)

print("---------")
print(min)
print(max)
print(avg/it)

plt.plot(x,y)
plt.show()
