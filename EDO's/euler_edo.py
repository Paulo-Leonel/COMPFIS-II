import numpy as np
import matplotlib.pyplot as plt
from math import sin, cos


# Graph parameters
plt.rcParams["xtick.labelsize"] = 14
plt.rcParams["ytick.labelsize"] = 14
plt.rcParams["axes.labelsize"] = 16


def main():
    
    # Constants
    a = 0.0  # Start of the interval
    b = 10.0  # Finish of the interval
    N = 10  # Steps by Euler solution
    h = (b - a) / N  # Size of one step of the solution
    x = 0.0  # Initial condition, x(a)

    exact_N = int(1e3)  # Number of points for the exact solution
    exact_h = (b - a) / exact_N

    # Exact solution
    exact_x = []
    time = np.arange(a, b, exact_h)
    for t in time:
        exact_x.append(cos(a) + x - cos(t))
    
    # Euler solution
    x_euler = []
    t_euler = np.arange(a, b, h)
    for t in t_euler:
        x_euler.append(x)
        x += h * f(x, t)
    
    # Graph
    plt.figure(figsize=(12,12))
    plt.plot(t_euler, x_euler, 'r.', time, exact_x)
    plt.xlabel("t")
    plt.ylabel("x(t)")
    plt.show()


def f(x, t):
    return sin(t)


if __name__ == '__main__':
    main()
