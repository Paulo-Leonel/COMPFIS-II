"""
    Solving the system of three differential equations, the
    Lorenz attractor.
"""

import numpy as np
import matplotlib.pyplot as plt

# Graph parameters
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['lines.linewidth'] = 1

# Global constants
sigma = 10.0
p = 28.0  # r constant
b = 8.0 / 3.0


def main():
    
    # Constants
    a = 0.0  # Start the interval of the independent variable
    b = 50.0  # Finish the interval of the independent variable
    N = 5000  # Steps for numerical integration
    h = (b - a) / N  # Size of one step

    time = np.arange(a, b, h)
    x_rk4, y_rk4, z_rk4 = [], [], []

    ra = np.array([0.0, 1.0, 0.0], float)
    r = ra

    for t in time:
        x_rk4.append(r[0])
        y_rk4.append(r[1])
        z_rk4.append(r[2])
        r += Runge_Kutta4(f,r,t,h)

    # Plot of y variable
    plt.figure(figsize=(12,12))
    plt.plot(time, y_rk4)
    plt.xlabel('Time (t)')
    plt.ylabel('y(t)')
    plt.show()

    # Plot of the Lorenz attractor
    with plt.style.context('dark_background'):
        plt.figure(figsize=(12,12))
        plt.plot(x_rk4, z_rk4, 'y-')
        plt.xlabel('x(t)')
        plt.ylabel('z(t)')
        plt.title("Lorenz attractor", fontsize=14)
    plt.show()


#-------------------------------------------

def f(r, t):
    x, y, z = r[0], r[1], r[2]
    fx, fy, fz = sigma*(y - x), x*(p-z)-y, x*y-b*z
    return np.array([fx, fy, fz], float)


def Runge_Kutta4(f, r, t, h):
    k1 = h * f(r, t)
    k2 = h * f(r + 0.5*k1, t + 0.5*h)
    k3 = h * f(r + 0.5*k2, t + 0.5*h)
    k4 = h * f(r + k3, t + h)
    return (k1 + 2.0*(k2 + k3) + k4) / 6.0


#-------------------------------------------

if __name__ == '__main__':
    main()
