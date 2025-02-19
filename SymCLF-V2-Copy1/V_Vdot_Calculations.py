# Compute V, V_dot, and related terms numerically
import numpy as np
from SystemDynamics import f, G, Q, R # Import system dynamics functions
from Functions import safe_divide # Import a safe division utility function

def compute_v_and_v_dot(V_function, x1_vals, x2_vals):
    """
    Computes the value function V, its gradient, and related control terms numerically.

    Parameters:
        V_function (ndarray): A 2D array representing the value function over the state space.
        x1_vals (ndarray): 1D array of x1 values (state variable 1).
        x2_vals (ndarray): 1D array of x2 values (state variable 2).

    Returns:
        tuple: Contains computed values for:
            - V_vals: The value function
            - V_grad_mag: Magnitude of the gradient of V
            - V_dot_vals: Time derivative of V
            - lambda_x_vals: Computed lambda values
            - u_vals[..., 1]: Control input component
            - u_vals2[..., 1]: Alternative control input component
            - u_mag: Control effort magnitude
            - a: Dot product of V gradient and system dynamics
    """

    # Create a grid of state values
    X1, X2 = np.meshgrid(x1_vals, x2_vals)
    
    # Evaluate system dynamics at each grid point
    f_vector = f(X1, X2) # System dynamics function
    G_matrix = G(X1, X2) # Control influence matrix
    Q_matrix = Q(X1, X2) # State cost weighting matrix
    R_matrix = R(X1, X2) # Control cost weighting matrix
    
    # Evaluate V
    V_vals = V_function  # Numerical function
    
    # Compute numerical gradients
    V_grad_x1 = np.gradient(V_vals, x1_vals, axis=1)  # Gradient w.r.t x1
    V_grad_x2 = np.gradient(V_vals, x2_vals, axis=0)  # Gradient w.r.t x2
    V_grad = np.stack([V_grad_x1, V_grad_x2], axis=-1)  # Combine gradients into vector field
    V_grad_mag = (V_grad_x1**2 + V_grad_x2**2)**0.5
 
    # Compute control terms
    b_T = np.einsum('ijk,kl->ijl', V_grad, G_matrix)
    a = np.einsum('ijk,kij->ij', V_grad, f_vector)

    # Compute norms
    x_stack = np.dstack([X1, X2])  # Combine X1 and X2 into a 3D array
    x_norm_squared = np.einsum('ijk,kl,ijl->ij', x_stack, Q_matrix, x_stack)  # Quadratic form
    b_norm_squared = np.einsum('ijk,kl,ijl->ij', b_T, np.linalg.inv(R_matrix), b_T)

    # Compute lambda_x values using a safe division function
    lambda_x_vals = safe_divide(
        np.sqrt(a ** 2 + x_norm_squared * b_norm_squared) + a,
        b_norm_squared,
        default=0
    )

    # Compute optimal control input u
    u_vals = -np.einsum('kl,ijk,ij->ijk', np.linalg.inv(R_matrix), b_T, lambda_x_vals)

    # Apply masking: Set u_vals to zero where b_T is zero
    u_vals[b_norm_squared < 0.000001] = 0

    # Compute an alternative control input without lambda_x scaling
    u_vals2 = -np.einsum('kl,ijk->ijk', np.linalg.inv(R_matrix), b_T)

    # Compute the time derivative of the value function (V_dot)
    V_dot_vals = np.einsum('ijk,ijk->ij', V_grad, f_vector.transpose(1, 2, 0)) + \
            np.einsum('ijk,ijk->ij', V_grad, np.einsum('kl,ijk->ijk', G_matrix, u_vals))

    # Compute control effort magnitude
    u_mag = np.einsum('ijk,kl,ijk->ij', u_vals, R_matrix, u_vals)

    # Return computed results
    return V_vals, V_grad_mag, V_dot_vals, lambda_x_vals, u_vals[..., 1], u_vals2[..., 1], u_mag, a, b_norm_squared