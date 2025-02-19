import numpy as np
from scipy.integrate import solve_ivp
from sympy import symbols, Matrix, sin, cos, lambdify, tanh
import matplotlib.pyplot as plt
from ControllerComp import f, G, Q, R

def NumSol(u_func, T, x1_0, x2_0, b_norm_squared):
    
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

    if b_norm_squared is not None:        
        b_norm_squared_func = lambdify((x1, x2), b_norm_squared, 'numpy')

    # Storage list for u_val
    #u_val_list = []
    
    # Define the dynamics for numerical integration
    def dynamics(t, x):
        x1_val, x2_val = x
        f_val = np.array(f_func(x1_val, x2_val)).flatten()  # Evaluate f(x1, x2)
        G_val = np.array(G_func(x1_val, x2_val))  # Evaluate G(x1, x2)
        u_val = np.array(u_func(x1_val, x2_val)).flatten()  # Evaluate u(x1, x2)

        if b_norm_squared is not None:
            
            b_norm_squared_val = b_norm_squared_func(x1_val, x2_val)
            
            #Compute the condition
            condition = b_norm_squared_val < 0.000001
            # Apply the condition: Set u_val to 0 where the condition is met
            u_val[condition] = 0

        # Store u_val for output
        #u_val_list.append(u_val)

        # Apply the upper bound to u
        u_val = np.clip(u_val, -1000, 1000)  # Limit u between -10 and 10
        
        return f_val + G_val @ u_val  # Compute \dot{x} = f + G * u

    # Initial conditions
    x1_0, x2_0 = x1_0, x2_0 # Initial state values
    t_span = (0, T)  # Time range 
    t_eval = np.linspace(0, T, 500)  # Time points for evaluation

    # Storage for u values at the selected `t_eval` points
    #u_valss = []
    '''
    # Function to record `u_val` at fixed time steps
    def store_control(t, x):
        x1_val, x2_val = x
        u_valsd = np.array(u_func(x1_val, x2_val)).flatten()
        u_valss.append(u_valsd)
    '''
    # Solve the ODE numerically
    solution = solve_ivp(dynamics, t_span, [x1_0, x2_0], t_eval=t_eval)
    #solve_ivp(dynamics, t_span, [x1_0, x2_0], t_eval=t_eval, vectorized=True, events=store_control)

    # Extract the results
    t_vals = solution.t
    x1_vals = solution.y[0]
    x2_vals = solution.y[1]

    # Convert u_val_list to a NumPy array
    #u_val = np.array(u_vals)

    return t_vals, x1_vals, x2_vals#, u_val