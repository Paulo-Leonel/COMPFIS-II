import numpy as np
import matplotlib.pyplot as plt
from math import sin, cos, pi, sqrt

# Graph parameters
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['axes.labelsize'] = 14

# Global constants
g = 9.8  # Standard gravity value
L = 1.0  # Wire length
theta_0 = 17*pi/18  # Initial angle
omega_0 = 0.0  # Initial angular velocity


def main():

    # Constants
    a = 0.0  # Start the interval
    b = 10.0  # Finish the interval
    N = 1000  # Steps for numerical integration
    h = (b - a) / N  # Size of one step

    time = np.arange(a, b, h)
    theta_rk4, omega_rk4 = [], []

    r_0 = np.array([theta_0, omega_0], float)  # Initial condition F(a)
    r = r_0
    for t in time:  # Numerical integration
        theta_rk4.append(r[0])
        omega_rk4.append(r[1])
        r += Runge_Kutta4(f, r, t, h)

    # Exact solution for comparison
    theta_e = []
    omega = np.sqrt(g / L)
    a = theta_0
    b = omega_0 / omega
    for t in time:
        theta_e.append(a*cos(omega*t) + b*sin(omega*t))
    
    # Graph
    plt.figure(figsize=(12,12))
    plt.plot(time, theta_rk4, label='Numrical solution')
    plt.plot(time, theta_e, label='Exact solution')
    plt.legend(loc='best', fontsize='large')
    plt.xlabel('t')
    plt.ylabel('theta(t), omega(t)')
    plt.show()


#---------------------------------------------------

def f(r, t):
    theta, omega = r[0], r[1]
    f0, f1 = omega, -g/L * sin(theta)
    return np.array([f0, f1], float)


def Runge_Kutta4(f, r, t, h):
    k1 = h * f(r, t)
    k2 = h * f(r+0.5*k1, t+0.5*h)
    k3 = h * f(r+0.5*k2, t+0.5*h)
    k4 = h * f(r+k3, t+h)
    return (k1+ 2.0*(k2+k3) + k4) / 6.0


#---------------------------------------------------

if __name__ == '__main__':
    main()
