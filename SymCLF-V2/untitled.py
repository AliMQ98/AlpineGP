import numpy as np
from V_Vdot_Calculations import compute_v_and_v_dot

def one_minus_cos(x):
    """
    Computes 1 - cos(x).

    Parameters:
    x : float or numpy.ndarray
        Input value(s).

    Returns:
    float or numpy.ndarray
        Computed result of 1 - cos(x).
    """
    return 1 - np.cos(x)

def eval_MSE_sol(individual, true_data):
    """
    Evaluates the Mean Squared Error (MSE) of a given individual (control function)
    based on Lyapunov function conditions.

    Parameters:
    individual : function
        A callable function representing the candidate Lyapunov function.
    true_data : Dataset
        A dataset containing input values (state variables x1 and x2).

    Returns:
    tuple:
        - MSE : float
            The computed Mean Squared Error with penalties for violations.
        - V_vals : numpy.ndarray
            The evaluated Lyapunov function values over the state space.
    """
    try:
        # Extract state variables from the dataset
        x1_vals = true_data.get_input(0)  
        x2_vals = true_data.get_input(1)  

        # Create mesh grid for evaluation
        X1, X2 = np.meshgrid(x1_vals, x2_vals)

        # Ensure the individual function is callable
        if not callable(individual):
            return 1e10  # Assign high error to discard invalid individuals

        # Evaluate Lyapunov function over the state space
        V_vals = individual(X1, X2)
        V_val_0 = individual(0, 0)  # Evaluate V at the origin

        # Compute Lyapunov-related properties
        _, V_grad_mag, V_dot_vals, lambda_vals, u_vals, _, u_mag, a_vals, _ = compute_v_and_v_dot(V_vals, x1_vals, x2_vals)

        # Count violations of Lyapunov function conditions
        V_violations = np.count_nonzero(V_vals <= 0)  # V should be positive
        V_dot_violations = np.count_nonzero(V_dot_vals >= 0)  # V_dot should be negative
        a_violations = 0  # Placeholder for additional violation checks

        # Initialize penalty variable
        penalty = 0

        # Check for NaN or infinite values and add a large penalty if found
        if np.any(np.isnan(V_vals)) or np.any(np.isinf(V_vals)):
            penalty += 1e6  
        if np.any(np.isnan(a_vals)) or np.any(np.isinf(a_vals)):
            penalty += 1e6  
        if np.any(np.isnan(V_dot_vals)) or np.any(np.isinf(V_dot_vals)):
            penalty += 1e6  
        if np.any(np.isnan(lambda_vals)) or np.any(np.isinf(lambda_vals)):
            penalty += 1e6  

        # Compute penalties based on lambda and control behavior
        lambda_penalty = 10 * np.mean((lambda_vals - 50) ** 2)  
        u_penalty = np.count_nonzero(u_mag > 25) * 1  

        # Compute gradients of lambda and control input
        lambda_grad_x1 = np.gradient(lambda_vals, x1_vals, axis=1)  
        lambda_grad_x2 = np.gradient(lambda_vals, x2_vals, axis=0)  
        lambda_grad_mag = np.sqrt(lambda_grad_x1**2 + lambda_grad_x2**2)
        lambda_grad_mag = np.mean(lambda_grad_mag) * 0.5  

        u_grad_x1 = np.gradient(u_vals, x1_vals, axis=1)  
        u_grad_x2 = np.gradient(u_vals, x2_vals, axis=0)  
        u_grad_mag = np.sqrt(u_grad_x1**2 + u_grad_x2**2)
        u_grad_mag = np.mean(u_grad_mag) * 1  

        # Compute final MSE including violations and penalties
        MSE = (
            V_violations + a_violations + V_dot_violations + penalty
            + 1000 * V_val_0**2  # Penalize if V(0,0) is not close to zero
            + 0.1 * np.mean(V_grad_mag)  
            + 0.1 * lambda_penalty  
            + lambda_grad_mag  
        )

        # Handle invalid MSE values
        if np.isnan(MSE) or np.isinf(MSE):
            MSE = 1e10  # Assign a large penalty

        return MSE, V_vals

    except Exception as e:
        print(f"Error in eval_MSE_sol: {e}")
        return 1e10, None  # Assign a large penalty in case of failure
