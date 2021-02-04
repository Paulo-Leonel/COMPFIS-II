# Ordinary differential equation (ODE)

The methods used for numerical integration is Runge-Kutta.

* Euler (First-order)
* Runge-Kutta 2 (Second-order)
* Runge-Kutta 4 (fourth-order)

The main idea of the method is using a trial step at the midpoint of an interval to cancel out lower-order error terms. (For further information see in [link](https://mathworld.wolfram.com/Runge-KuttaMethod.html "Math World Wolfram"))

The Runge-methods is really simple and robust and is a good choice for solution of differential equations when combined with an adaptive step-size routine.

## The routine for each method is below

### Euler

```python
def Euler(f, x, t, h):
    k1 = h * f(x, t)
    return k1
```
### Runge-Kutta 2

```python
def Runge_Kutta2(f, x, t, h):
    k1 = h * f(x, t)
    k2 = h * f(x + 0.5*k1, t + 0.5*h)
    return k2
```     
### Runge-Kutta 4

```python
def Runge_Kutta4(f, x, t, h):
    k1 = h * f(x, t)
    k2 = h * f(x + 0.5*k1, t + 0.5*h)
    k3 = h * f(x + 0.5*k2, t + 0.5*h)
    k4 = h * f(x + k3, t + h)
    return (k1 + 2.0*(k2 + k3) + k4) / 6.0
```  
