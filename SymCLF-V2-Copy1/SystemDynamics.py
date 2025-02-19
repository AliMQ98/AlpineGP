# System Dynamics Model
import numpy as np

# Define the system dynamics numerically
def f(x1, x2):
    """
    Computes the system dynamics function f(x).

    Parameters:
        x1 (float or ndarray): State variable representing angular position.
        x2 (float or ndarray): State variable representing angular velocity.

    Returns:
        ndarray: A 2D array representing the time derivatives [dx1/dt, dx2/dt].
    
    The system represents a **damped inverted pendulum**, governed by:
        dx1/dt = x2
        dx2/dt = (g/l) * sin(x1) - (b / (m * l^2)) * x2
    """
    
    # Constants
        # Acceleration due to gravity (m/s^2)
        # Length of the pendulum (m)
        # Damping coefficient
        # Mass of the pendulum (kg)
    g, l, b, m = 9.81, 0.5, 0.1, 0.15 
    
    A, B = g / l, -b / (m * l**2) # Gravity term and Damping term

    # Return system dynamics as a 2D array
    return np.array([x2, A * np.sin(x1) + B * x2])

def G(x1, x2):
    """
    Defines the control influence matrix G(x), indicating how control inputs affect the system.
    
    Returns:
        ndarray: A (2×2) matrix defining control input influence.
    """
    return np.array([[0, 0], [0, 26.7]])

def Q(x1, x2):
    """
    Defines the state cost matrix Q(x) used in optimal control.
    
    Returns:
        ndarray: A (2×2) positive semi-definite matrix representing the weight on state errors.
    """
    return np.array([[1, 0], [0, 1]])

def R(x1, x2):
    """
    Defines the control cost matrix R(x) used in optimal control.
    
    Returns:
        ndarray: A (2×2) positive definite matrix representing the cost of control effort.
    """
    return np.array([[1, 0], [0, 1]])


