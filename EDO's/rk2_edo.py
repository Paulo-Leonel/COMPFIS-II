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
    N = 10  # Steps by Runge-Kutta (2º order)
    h = (b - a) / N  # Size of one step
    x = 0.0  # Initial condition, x(a)

    exact_N = int(1e3)  # Number of points for the exact solution
    exact_h = (b - a) / exact_N

    # Exact solution
    exact_x = []
    time = np.arange(a, b, exact_h)
    for t in time:
        exact_x.append(cos(a) + x - cos(t))
    
    # Runge-Kutta 2 solution
    t_rk2 = np.arange(a, b, h)
    x_rk2 = []
    for t in t_rk2:
        x_rk2.append(x)
        k1 = h * f(x, t)
        k2 = h * f(x + 0.5 * k1, t + 0.5 * h)
        x += k2
    
    # Graph
    plt.figure(figsize=(12,12))
    plt.plot(t_rk2, x_rk2, 'r.', time, exact_x)
    plt.xlabel("t")
    plt.ylabel("x(t)")
    plt.show()


def f(x, t):
    return sin(t)


if __name__ == "__main__":
    main()
