import numpy as np
from V_Vdot_Calculations import compute_v_and_v_dot

def one_minus_cos(x):
    return 1 - cos(x)

def eval_MSE_sol(individual, true_data):
    #warnings.filterwarnings("ignore")

    try:

            
        X1 = true_data.get_input(0)  # Access X1
        X2 = true_data.get_input(1)  # Access X2

        V_vals = individual(X1, X2)
        V_val_0 = individual(0, 0)

        #print(V_vals.shape)
        #V_vals = np.tile(V_vals[1, ...], (100, 1))
        # Construct the symmetric matrix using outer sum
        V_vals = np.add.outer(V_vals, V_vals) / 2
        #print(V_vals.shape)
        #print(V_vals[1, :])
        
        V_vals, V_grad_mag, V_dot_vals, lambda_vals, _, _, u_mag = compute_v_and_v_dot(V_vals, X1, X2)

        V_violations = np.count_nonzero(V_vals <= 0)
        V_dot_violations = np.count_nonzero(V_dot_vals >= 0)
    
        # Check for invalid values (e.g., NaN or inf) and add penalty
        penalty = 0
        if np.any(np.isnan(V_vals)) or np.any(np.isinf(V_vals)):
            penalty += 1e6  # Add large penalty for invalid V_vals
        if np.any(np.isnan(V_dot_vals)) or np.any(np.isinf(V_dot_vals)):
            penalty += 1e6  # Add large penalty for invalid V_dot_vals
        if np.any(np.isnan(lambda_vals)) or np.any(np.isinf(lambda_vals)):
            penalty += 1e6  # Add large penalty for invalid lambda_vals
        #MSE = np.mean(np.square(y_pred - true_data.y))

        #*lmbda_penalty = 1* np.mean((lambda_vals - 1)**2) #+ np.std(lambda_vals)**2

        '''
        penalty = np.count_nonzero(u_mag > 15) * 1000

        #
        lmbda_penalty = 0 #(np.count_nonzero(lambda_vals < 0.8) + np.count_nonzero(lambda_vals > 1.2)) * 100
        #0.1 * ((lambda_vals.mean() - 1)**2)
        #0.1 * (((lambda_vals.sum() - lambda_vals.size)/lambda_vals.size)**2)**0.5
        

        lambda_grad_x1 = np.gradient(lambda_vals, X1, axis=1)  # Gradient w.r.t x1        
        lambda_grad_x2 = np.gradient(lambda_vals, X2, axis=0)  # Gradient w.r.t x2
        lambda_grad_mag = (lambda_grad_x1**2 + lambda_grad_x2**2)**0.5
        lambda_grad_mag = np.mean(lambda_grad_mag) * 100
        '''

        
        MSE = (1000000000 * V_violations + V_dot_violations + penalty + V_val_0**2) + 0.05 * np.mean(V_grad_mag) #+ 1 * lmbda_penalty) #+ lambda_grad_mag\
        #+ 100 * np.mean(u_mag) #
        #+ (V_vals.sum()**2 + V_dot_vals.sum**2)**0.1
        
        if np.isnan(MSE):
            MSE = 1e10
        return MSE, V_vals
    except Exception:
        return 1e10, V_vals