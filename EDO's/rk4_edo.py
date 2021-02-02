import numpy as np
import matplotlib.pyplot as plt
from math import cos, sin, exp

# Global constants for the function
gama, omega = 1.0, 3.0

# Graph parameters
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14
plt.rcParams['axes.labelsize'] = 16

def main():

    # Constants
    a = 0  # Start the interval
    b = 10  # Finish the interval
    N = 8  # Steps for all algorithms
    h = (b - a) / N  # Size of one step
    xa = 0.0  # Inital condition, x(a)

    # Exact solution
    N_e = 1000  # Number of points in the exact solution
    h_e = (b - a) / N_e

    t_e = np.arange(a, b, h_e)
    x_e = []
    for t in t_e:
        x_e.append(exact_sol(t, a, xa))

    # Time for all
    time = np.arange(a, b, h)
    
    # Euler solution
    x_euler = []
    xe = xa
    for t in time:
        x_euler.append(xe)
        xe += Euler(f, xe, t, h)
    
    # Runge-Kutta 2
    x_rk2 = []
    xrk2 = xa
    for t in time:
        x_rk2.append(xrk2)
        xrk2 += Runge_Kutta2(f, xrk2, t, h)
    
    # Runge-Kutta 4 (4º order)
    x_rk4 = []
    xrk4 = xa
    for t in time:
        x_rk4.append(xrk4)
        xrk4 += Runge_Kutta4(f, xrk4, t, h)
    
    # Graph
    plt.figure(figsize=(12,12))
    plt.plot(time, x_euler, 'gs', label='Euler')
    plt.plot(time, x_rk2, 'yo', label='Runge-Kutta2')
    plt.plot(time, x_rk4, 'b.', label='Runge-Kutta4')
    plt.plot(t_e, x_e, label='Exact solution')
    plt.legend(loc='best', fontsize='large')
    plt.xlabel('t')
    plt.ylabel('x(t)')
    plt.show()


    # Analysing the error for many N values

    # Constants
    a = 0  # Start the interval
    b = 10  # Finish the interval
    N = []
    for i in range(3, 11):
        N.append(2**i)
    
    error_euler = []
    error_rk2 = []
    error_rk4 = []

    for n in N:

        # Initial values for the algorithms
        x = y = z = 0.0
        h = (b - a) / n

        time = np.arange(a, b, h)

        # Exact solution
        x_e = []
        for t in time:
            x_e.append(exact_sol(t, a, xa))

        # Euler
        x_euler = []
        for t in time:
            x_euler.append(x)
            x += Euler(f, x, t, h)
        
        # Runge-Kutta 2
        x_rk2 = []
        for t in time:
            x_rk2.append(y)
            y += Runge_Kutta2(f, y, t, h)
        
        # Runge-Kutta 4
        x_rk4 = []
        for t in time:
            x_rk4.append(z)
            z += Runge_Kutta4(f, z, t, h)
        
        # Error
        error_euler.append(np.sqrt(Error(x_euler, x_e) / n))
        error_rk2.append(np.sqrt(Error(x_rk2, x_e) / n))
        error_rk4.append(np.sqrt(Error(x_rk4, x_e) / n))
    
    # Graphing the error
    plt.figure(figsize=(12, 12))
    plt.loglog(N, error_euler, 'r.', label='Euler')
    plt.loglog(N, error_rk2, 'gs', label='Runge-Kutta 2')
    plt.loglog(N, error_rk4, 'bo', label='Runge-Kutta 4')
    plt.legend(loc='best', fontsize='large')
    plt.xlabel("N")
    plt.ylabel("Error")
    plt.show()


#--------------------------------
# Algorithms and functions

def f(x, t):
    return -2 * exp(-gama*t) * (gama*cos(omega*t) + omega*sin(omega*t))


def exact_sol(t, a, xa):
    b = xa - 2.0 * exp(-gama * a) * cos(omega * a)
    return 2.0 * exp(-gama * t) * cos(omega * t) + b


def Euler(f, x, t, h):
    return h * f(x, t)


def Runge_Kutta2(f, x, t, h):
    k1 = h * f(x, t)
    k2 = h * f(x + 0.5 * k1, t + 0.5 * h)
    return k2


def Runge_Kutta4(f, x, t, h):
    k1 = h * f(x, t)
    k2 = h * f(x + 0.5 * k1, t + 0.5 * h)
    k3 = h * f(x + 0.5 * k2, t + 0.5 * h)
    k4 = h * f(x + k3, t + h)
    return (k1 + 2.0*(k2 + k3) + k4) / 6.0


def Error(x_alg, x_exact):
    error = 0.0
    for j in range(len(x_exact)):
        error += (x_alg[j] - x_exact[j]) ** 2
    return error


#-----------------------------------------------------

if __name__ == '__main__':
    main()
