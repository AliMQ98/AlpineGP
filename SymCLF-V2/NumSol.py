import numpy as np
from scipy.integrate import solve_ivp
from sympy import symbols, Matrix, sin, cos, lambdify, tanh
import matplotlib.pyplot as plt
from ControllerComp import f, G, Q, R # Import symbolic functions f and G

def NumSol(u_func, T, x1_0, x2_0, b_norm_squared):
    """
    Numerically solves a system of differential equations with control input.

    Parameters:
    u_func : function
        Control input function u(x1, x2).
    T : float
        Total simulation time.
    x1_0 : float
        Initial condition for state variable x1.
    x2_0 : float
        Initial condition for state variable x2.
    b_norm_squared : function or None
        b_norm_squared!!!

    Returns:
    t_vals : numpy.ndarray
        Array of time values.
    x1_vals : numpy.ndarray
        Array of state variable x1 values over time.
    x2_vals : numpy.ndarray
        Array of state variable x2 values over time.
    """
    
    # Define symbolic variables for time and state
    t = symbols('t')  # Time
    x1, x2 = symbols('x1 x2')  # State variables

    # Define symbolic f(x1, x2), G(x1, x2), and u(x1, x2)
    f_vector = f(x1, x2)
    G_matrix = G(x1, x2)

    u = Matrix([0, u_func])  # Control input
 
    # Convert symbolic expressions to numerical functions for computation
    f_func = lambdify((x1, x2), f_vector, 'numpy')  # f(x1, x2) as a NumPy function
    G_func = lambdify((x1, x2), G_matrix, 'numpy')  # G(x1, x2) as a NumPy function
    u_func = lambdify((x1, x2), u, 'numpy')  # u(x1, x2) as a NumPy function

    # Convert b_norm_squared to a numerical function if provided
    if b_norm_squared is not None:        
        b_norm_squared_func = lambdify((x1, x2), b_norm_squared, 'numpy')
    
    # Define the dynamics for numerical integration
    def dynamics(t, x):
        """
        Computes the time derivative of the state vector.
        
        Parameters:
        t : float
            Current time.
        x : list or numpy.ndarray
            Current state values [x1, x2].

        Returns:
        numpy.ndarray
            The time derivative of the state vector [dx1/dt, dx2/dt].
        """

        # Extract state variables
        x1_val, x2_val = x

        # Introduce small random noise to the system (currently set to zero)
        noise = (-1 + 2 * np.random.rand(2)) * 0  # Generates a 2-element noise array
        x1_val, x2_val = x1_val * (1 + noise[0]), x2_val * (1 + noise[1])

        # Evaluate system dynamics
        f_val = np.array(f_func(x1_val, x2_val)).flatten()  # Evaluate f(x1, x2)
        G_val = np.array(G_func(x1_val, x2_val))  # Evaluate G(x1, x2)
        u_val = np.array(u_func(x1_val, x2_val)).flatten()  # Evaluate u(x1, x2)

        # Check if b_norm_squared is provided and enforce control constraints
        if b_norm_squared is not None:
            b_norm_squared_val = b_norm_squared_func(x1_val, x2_val)
            
            # Compute the condition
            condition = b_norm_squared_val < 0.000001
            # Apply the condition: Set u_val to 0 where the condition is met
            u_val[condition] = 0

        # Apply the upper bound to u
        u_val = np.clip(u_val, -1000, 1000)  # Limit u between -10 and 10
        
        return f_val + G_val @ u_val  # Compute \dot{x} = f + G * u

    # Define initial conditions and time span
    x1_0, x2_0 = x1_0, x2_0 # Initial state values
    t_span = (0, T)  # Time range 
    t_eval = np.linspace(0, T, 500)  # Time points for evaluation

    # Solve the system of ODEs using numerical integration
    solution = solve_ivp(dynamics, t_span, [x1_0, x2_0], t_eval=t_eval)
    #solve_ivp(dynamics, t_span, [x1_0, x2_0], t_eval=t_eval, vectorized=True, events=store_control)

    # Extract the results
    t_vals = solution.t
    x1_vals = solution.y[0]
    x2_vals = solution.y[1]

    return t_vals, x1_vals, x2_vals