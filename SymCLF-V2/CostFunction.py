# Cost Function 
import numpy as np
from ControllerComp import f, G, Q, R
from sympy import symbols, Matrix, sin, cos, lambdify, tanh

def calculate_cost(t_vals, x1_vals, x2_vals, u_func, b_norm_squared, skip_steps):
    """
    Calculate the cost function J = ∫ (x^T Q x + u^T R u) dt.

    This function computes the total cost based on state and control trajectories
    over a given time horizon.

    Parameters:
    t_vals : numpy.ndarray
        Array of time values.
    x1_vals : numpy.ndarray
        Array of x1 state values over time.
    x2_vals : numpy.ndarray
        Array of x2 state values over time.
    u_func : function
        Control input function u(x1, x2).
    b_norm_squared : function or None
        A function that determines if the control input should be deactivated based on its value.
    skip_steps : int
        Number of initial time steps to skip in the cost computation.

    Returns:
    total_cost : float
        The computed total cost J.
    total_cost1 : float
        The cost contribution from state-related terms (x^T Q x).
    total_cost2 : float
        The cost contribution from control-related terms (u^T R u).
    u_val : numpy.ndarray
        Array of evaluated control inputs over time.
    """
    
    # Define system constants
    g, l, b, m = 9.81, 0.5, 0.1, 0.15

    # Define symbolic state variables
    x1, x2 = symbols('x1 x2')  # State variables

    # Define symbolic control input and convert to numerical function
    u = Matrix([0, u_func])  # Control input
    u = lambdify((x1, x2), u, 'numpy')

     # Convert b_norm_squared to a numerical function if provided
    if b_norm_squared is not None:        
        b_norm_squared_func = lambdify((x1, x2), b_norm_squared, 'numpy')

    # Storage list for u_val
    u_val_list = []

    # Initialize cost
    total_cost = 0
    total_cost1 = 0
    total_cost2 = 0

    # Time step
    dt = t_vals[1] - t_vals[0]

    # Loop through each time step
    for i, (t, x1, x2) in enumerate(zip(t_vals, x1_vals, x2_vals)):
        if i < skip_steps:
            continue  # Skip the first `skip_steps` steps
            
        try: 
            
            # State vector
            x = np.array([x1, x2])

            # Evaluate Q(x1, x2) and R(x1, x2)
            Q_val = np.array(Q(x1, x2), dtype=float)
            R_val = np.array(R(x1, x2), dtype=float)

            # Control input (evaluate u_func)
            u_val = u(x1, x2)  # u_func must return a scalar or vector

            # Apply condition from b_norm_squared if provided
            if b_norm_squared is not None:
                b_norm_squared_val = b_norm_squared_func(x1, x2)
                #Compute the condition
                condition = b_norm_squared_val < 0.000001
                # Apply the condition: Set u_val to 0 where the condition is met
                u_val[condition] = 0

            #Store u_val for output
            u_val = np.clip(u_val, -1000, 1000)
            
            u_val_list.append(u_val)

            # Quadratic cost at this time step
            cost1 = x.T @ Q_val @ x 
            cost2 = u_val.T @ R_val @ u_val * (m * l**2)**2
            cost = cost1 + cost2
    
            # Accumulate the cost (approximation using Riemann sum)
            total_cost1 += cost1 * dt
            total_cost2 += cost2 * dt
            total_cost += cost * dt
            
        except Exception as e:
            # Skip this time step if u cannot be computed
            print(f"Skipping t={t}, x1={x1}, x2={x2} due to: {e}")
            continue

        u_val = np.array(u_val_list)  # Convert list to NumPy array        

    return total_cost, total_cost1, total_cost2, u_val
