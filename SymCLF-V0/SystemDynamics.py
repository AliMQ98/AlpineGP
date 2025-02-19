# System Dynamics
import numpy as np

# Define the system dynamics numerically
def f(x1, x2):
    # Constants
    g, l, b, m = 9.81, 0.5, 0.1, 0.15
    A, B = -g / l, -b / (m * l**2)
    return np.array([x2, A * np.sin(x1) + B * x2])

def G(x1, x2):
    return np.array([[0, 0], [0, 1]])

def Q(x1, x2):
    return np.array([[1, 0], [0, 1]])

def R(x1, x2):
    return np.array([[1, 0], [0, 1]])


