import numpy as np
from scipy.integrate import solve_ivp
from sympy import symbols, Matrix, sin, cos, lambdify, tanh
import matplotlib.pyplot as plt
from ControllerComp import f, G, Q, R

def NumSol(u_func, T, x1_0, x2_0):
    
        # Define symbolic variables
    t = symbols('t')  # Time
    x1, x2 = symbols('x1 x2')  # State variables

    # Define symbolic f(x1, x2), G(x1, x2), and u(x1, x2)
    f_vector = f(x1, x2)
    G_matrix = G(x1, x2)



    u = Matrix([0, u_func])  # Control input
 
    # Convert symbolic expressions to numerical functions
    f_func = lambdify((x1, x2), f_vector, 'numpy')  # f(x1, x2) as a NumPy function
    G_func = lambdify((x1, x2), G_matrix, 'numpy')  # G(x1, x2) as a NumPy function

    #u_func = sp.Piecewise((u, x2 != 0), (0, True)) 
    u_func = lambdify((x1, x2), u, 'numpy')  # u(x1, x2) as a NumPy function

    # Define the dynamics for numerical integration
    def dynamics(t, x):
        x1_val, x2_val = x
        f_val = np.array(f_func(x1_val, x2_val)).flatten()  # Evaluate f(x1, x2)
        G_val = np.array(G_func(x1_val, x2_val))  # Evaluate G(x1, x2)
        u_val = np.array(u_func(x1_val, x2_val)).flatten()  # Evaluate u(x1, x2)

        # Compute the condition
        #condition = (78 * x1_val + 36 * x2_val) < 1
        # Apply the condition: Set u_val to 0 where the condition is met
        #u_val[condition] = 0

        # Apply the upper bound to u
        u_val = np.clip(u_val, -1000, 1000)  # Limit u between -10 and 10
        
        return f_val + G_val @ u_val  # Compute \dot{x} = f + G * u

    # Initial conditions
    x1_0, x2_0 = x1_0, x2_0 # Initial state values
    t_span = (0, T)  # Time range 
    t_eval = np.linspace(0, T, 500)  # Time points for evaluation

    # Solve the ODE numerically
    solution = solve_ivp(dynamics, t_span, [x1_0, x2_0], t_eval=t_eval)

    # Extract the results
    t_vals = solution.t
    x1_vals = solution.y[0]
    x2_vals = solution.y[1]

    return t_vals, x1_vals, x2_vals