#from .ornstein_uhlenbeck import OrnsteinUhlenbeckActionNoise
import numpy as np
import math
from sympy.solvers import solve
from sympy import Symbol
from sympy import sin as symsin

STEP_SIZE = 5
BRIDGE_WIDTH = 160


def get_optimal_step(x_pos, y_pos, commitment=1):
    a = Symbol("a")
    if y_pos == 0:
        sol_angle = 0
    elif -STEP_SIZE < y_pos < STEP_SIZE:
        angle = solve(symsin(a) * 5 + y_pos)
        sol_angle_list = [math.degrees(x) if -90 < math.degrees(x) < 90 else None for x in angle]
        try:
            sol_angle = next(item for item in sol_angle_list if item is not None)
        except StopIteration as sie:
            print(sie)
            sol_angle = 0
    elif y_pos < 0:
        sol_angle = 90
    elif y_pos > 0:
        sol_angle = -90

    y_pos = y_pos + math.sin(sol_angle) * STEP_SIZE
    x_pos = x_pos + math.cos(sol_angle) * STEP_SIZE

    return sol_angle, x_pos, y_pos