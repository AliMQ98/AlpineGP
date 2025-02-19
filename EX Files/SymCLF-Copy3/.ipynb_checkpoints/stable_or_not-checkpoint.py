from sympy import symbols, Function, Matrix, Eq, dsolve, lambdify, diff, Matrix, sin, cos, exp, log, sqrt, tanh
from ControllerComp import V_x, f, x, G, Q, R, safe_divideSR, compute_v_and_v_dot2
from NumSol import NumSol
from SymFunctions import read_expression, DeapSimplifier

import timeout_decorator

def stable_or_not(individuals_str):
    MSEE = {}  # Initialize dictionary to store results

    for i in range(len(individuals_str)):  # Fix loop iteration
        simplified_expr = DeapSimplifier(individuals_str[i])
        V = simplified_expr
        x1, x2 = symbols('x1 x2')
        V_grad = V_x(V, x1, x2)
        x_vector = x(x1, x2)
        f_vector = f(x1, x2)
        G_matrix = G(x1, x2)
        Q_matrix = Q(x1, x2)
        R_matrix = R(x1, x2)
        
        _, _, _, u_func, _, _, _ = compute_v_and_v_dot2(V)
        
        x1, x2 = symbols('x1 x2')
        
        T = 4
        x1_0, x2_0 = 1, 1
        
        try:
            @timeout_decorator.timeout(5)  # If computation takes >10 sec, skip
            def compute_trajectory():
                return NumSol(u_func, T, x1_0, x2_0, None)
            
            t_vals, x1_vals, x2_vals = compute_trajectory()

            if (x2_vals[-1]**2 + x1_vals[-1]**2)**0.5 > 0.01:
                MSEE[i] = 10e10  # Large error value for unstable cases
        
        except timeout_decorator.TimeoutError:
            MSEE[i] = float("inf")  # Assign infinite error if computation is too slow

    return MSEE

