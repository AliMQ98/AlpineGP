# Cost Function 
import numpy as np
from ControllerComp import f, G, Q, R
from sympy import symbols, Matrix, sin, cos, lambdify, tanh

def calculate_cost(t_vals, x1_vals, x2_vals, u_func, u_val, skip_steps):
    """
    Calculate the cost function J = ∫ (x^T Q x + u^T R u) dt
    """

    x1, x2 = symbols('x1 x2')  # State variables
    #u = Matrix([0, u_func])  # Control input
    #u = lambdify((x1, x2), u, 'numpy')

    # Initialize cost
    total_cost = 0
    total_cost1 = 0
    total_cost2 = 0

    # Time step
    dt = t_vals[1] - t_vals[0]

    # Loop through each time step
    for i, (t, x1, x2, u) in enumerate(zip(t_vals, x1_vals, x2_vals, u_val)):
        if i < skip_steps:
            continue  # Skip the first `skip_steps` steps
            
        try: 
            
            # State vector
            x = np.array([x1, x2])

            # Evaluate Q(x1, x2) and R(x1, x2)
            Q_val = np.array(Q(x1, x2), dtype=float)
            R_val = np.array(R(x1, x2), dtype=float)

            # Control input (evaluate u_func)
            u_val = u #(x1, x2)  # u_func must return a scalar or vector
            #u_val = np.clip(u_val, -20, 20)

            # Quadratic cost at this time step
            cost1 = x.T @ Q_val @ x 
            cost2 = u_val.T @ R_val @ u_val
            cost = cost1 + cost2
    
            # Accumulate the cost (approximation using Riemann sum)
            total_cost1 += cost1 * dt
            total_cost2 += cost2 * dt
            total_cost += cost * dt
            
        except Exception as e:
            # Skip this time step if u cannot be computed
            print(f"Skipping t={t}, x1={x1}, x2={x2} due to: {e}")
            continue

    return total_cost, total_cost1, total_cost2
