"""
    Study of the evolution of rabbit and fox population on an
    isolated ecosystem.
"""

import numpy as np
import matplotlib.pyplot as plt

# Graph parameters
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14
plt.rcParams['axes.labelsize'] = 16

# Global constants
alpha = 1.0
betha = gama = 0.5
delta = 2.0

def main():
    
    # Constants
    a = 0.0  # Start the interval of the independent variable
    b = 30.0  # Finish the interval of the independent variable
    N = int(1e3)  # Steps for numerical integration
    h = (b - a) / N  # Size of one step

    time = np.arange(a, b, h)
    x_rabbit, y_fox = [], []  # store population of rabbits and foxs

    ra = np.array([2.0,2.0],float)
    r = ra

    for t in time:
        x_rabbit.append(r[0])
        y_fox.append(r[1])
        r += Runge_Kutta4(f, r, t, h)
    

    # Graph
    plt.figure(figsize=(12,12))
    plt.plot(time, x_rabbit, 'b-', label='Rabbit population')
    plt.plot(time, y_fox, 'y-', label='Fox population')
    plt.legend(loc='best', fontsize='medium')
    plt.ylabel('rabbit(t), fox(t)')
    plt.xlabel("Time (t)")
    plt.show()


#-------------------------------------------------------

def f(r, t):
    x, y = r[0], r[1]
    fx, fy = alpha*x - betha*x*y, gama*x*y - delta*y
    return np.array([fx, fy], float)


def Runge_Kutta4(f, r, t, h):
    k1 = h * f(r, t)
    k2 = h * f(r + 0.5*k1, t + 0.5*h)
    k3 = h * f(r + 0.5*k2, t + 0.5*h)
    k4 = h * f(r + k3, t + h)
    return (k1 + 2.0*(k2 + k3) + k4) / 6.0


#-------------------------------------------------------

if __name__ == '__main__':
    main()
