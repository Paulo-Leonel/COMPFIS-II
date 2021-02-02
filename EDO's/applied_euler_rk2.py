import numpy as np
import matplotlib.pyplot as plt
from math import sin, cos, exp, sqrt, log

# Graph parameters
plt.rcParams["xtick.labelsize"] = 14
plt.rcParams["ytick.labelsize"] = 14
plt.rcParams["axes.labelsize"] = 16


def main():

    # Constants
    a = 0.0  # Start of the interval
    b = 10.0  # Finish of the interval
    xa = 0.0  # Initial condition, x(a)
    exact_N = int(1e3)  # Number of points for the exact solution
    exact_h = (b - a) / exact_N

    c = 1.0  # Constant for the function

    # Exact solution
    exact_x = []
    time = np.arange(a, b, exact_h)
    for t in time:
        exact_x.append(exact_sol(t, a, xa, c))

    # Euler and Runge-kutta 2 solution
    N_list = []
    for i in range(3, 11):
        N_list.append(2**i)
    
    meanerror_rk2 = []
    meanerror_euler = []

    for N in N_list:  # Number of steps for rk2
        h = (b - a) / N  # Size of one step
        x = xa

        t_rk2 = np.arange(a, b, h)
        x_rk2 = []
        x_euler = []
        xe = x
        error_rk2 = 0.0
        error_euler = 0.0

        for t in t_rk2:

            error_rk2 += (x - exact_sol(t,a,xa,c)) ** 2  # Accumulating the error of Runge-Kutta 2
            error_euler += (xe - exact_sol(t,a,xa,c)) ** 2  # Accumulating the error of Euler

            x_rk2.append(x)
            k1 = h * f(x, t, c)
            k2 = h * f(x + 0.5 * k1, t + 0.5 * h, c)
            x += k2
            x_euler.append(xe)
            xe += k1

        if N == 128:
            plt.figure(figsize=(12,12))
            plt.plot(t_rk2, x_rk2, 'gs', label="Runge-Kutta2")
            plt.plot(t_rk2, x_euler, 'ro', label="Euler")
            plt.plot(time, exact_x, label="Exact solution")
            plt.xlabel("t")
            plt.ylabel("x(t)")
            plt.legend(loc='best', fontsize='xx-large')
            plt.show()

        error_rk2 /= len(t_rk2)
        error_euler /= len(t_rk2)
        meanerror_rk2.append(np.sqrt(error_rk2))
        meanerror_euler.append(np.sqrt(error_euler))

    # Graph
    plt.figure(figsize=(12,12))
    plt.loglog(N_list, meanerror_rk2, 'gs', label="Runge-Kutta2")
    plt.loglog(N_list, meanerror_euler, 'ro', label="Euler")
    plt.xlabel("N")
    plt.ylabel("Mean Error")
    plt.legend(loc='best', fontsize='x-large')
    plt.show()

    print((log(meanerror_euler[0])-log(meanerror_euler[len(N_list)-1]))/(log(1024.0)-log(8.0)))
    print((log(meanerror_rk2[0])-log(meanerror_rk2[len(N_list)-1]))/(log(1024.0)-log(8.0)))


def exact_sol(t, a, xa, c):
    b = xa - 2.0 * exp(-c*a) * cos(a)
    return 2.0 * exp(-c*t) * cos(t) + b


def f(x, t, c):
    return -2.0 * exp(-c*t) * (c * cos(t) + sin(t))


if __name__ == "__main__":
    main()
