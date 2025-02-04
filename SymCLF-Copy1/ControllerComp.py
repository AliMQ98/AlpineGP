from sympy import symbols, lambdify, diff, Matrix, sin, cos, exp, log, sqrt, tanh, lambdify
from Functions import safe_divideSR

# Define symbolic variables
x1, x2 = symbols('x1 x2')

# Define x(x) as a vector
def x(x1, x2):
    return Matrix([x1, x2])

# Define f(x) as a vector
def f(x1, x2):
    g, l, b, m = 9.81, 0.5, 0.1, 0.15
    A, B = -g / l, -b / (m * l**2)
    
    f1 = x2
    f2 = A * sin(x1) + B * x2
    return Matrix([f1, f2])

# Define G(x) as a matrix
def G(x1, x2):
    G1 = [0, 0]
    G2 = [0, 1]
    return Matrix([G1, G2])

# Define Q(x) as a matrix
def Q(x1, x2):
    Q1 = [1, 0]
    Q2 = [0, 1]
    return Matrix([Q1, Q2])

# Define R(x) as a matrix
def R(x1, x2):
    R1 = [1, 0]
    R2 = [0, 1]
    return Matrix([R1, R2])

# Define the gradient of V(x1, x2)
def V_x(V, x1, x2):
    Vx1 = diff(V, x1)  # Partial derivative with respect to x1
    Vx2 = diff(V, x2)  # Partial derivative with respect to x2
    return Matrix([Vx1, Vx2])  # Return as a column vector

# Compute V and V_dot
def compute_v_and_v_dot2(V_value):
    f_vector = f(x1, x2)
    G_matrix = G(x1, x2)
    Q_matrix = Q(x1, x2)
    R_matrix = R(x1, x2)
    
    V_grad = Matrix([V_value.diff(x1), V_value.diff(x2)])
    b_T = V_grad.T * G_matrix
    a = V_grad.T * f_vector
    b_norm_squared = b_T * R_matrix.inv() * b_T.T
    x_norm_squared = Matrix([[x1, x2]]) * Q_matrix * Matrix([[x1], [x2]])
    
    #lambda_x = (sqrt(a[0] ** 2 + x_norm_squared[0] * b_norm_squared[0]) + a[0]) / b_norm_squared[0]
    '''
    lambda_x = safe_divideSR(
        sqrt(a[0] ** 2 + x_norm_squared[0] * b_norm_squared[0]) + a[0],
        b_norm_squared[0],
        default=0
    )
    '''
    lambda_x = sqrt(a[0] ** 2 + x_norm_squared[0] * b_norm_squared[0]) + a[0] / b_norm_squared[0]
    
    lambda_x_diff = lambda_x.diff(x1) + lambda_x.diff(x2)
    
    u_value = -R_matrix.inv() * b_T.T * lambda_x
    u_value2 = -R_matrix.inv() * b_T.T 
    
    V_dot = V_grad.T * f_vector + V_grad.T * G_matrix * u_value
    u_mag = u_value.T * R_matrix * u_value
    #return lambdify((x1, x2), V_value, 'numpy'), lambdify((x1, x2), V_dot[0], 'numpy'), lambdify((x1, x2), lambda_x, 'numpy'), lambdify((x1, x2), u_value[1], 'numpy'), lambdify((x1, x2), u_mag, 'numpy'), lambdify((x1, x2), lambda_x_diff, 'numpy')
    return V_value, V_dot[0], lambda_x, u_value[1], u_value2[1], u_mag

# Compute V and V_dot
def compute_v_and_v_dot3(V_value, u_func):
    f_vector = f(x1, x2)
    G_matrix = G(x1, x2)
    Q_matrix = Q(x1, x2)
    R_matrix = R(x1, x2)
    
    V_grad = Matrix([V_value.diff(x1), V_value.diff(x2)])
    b_T = V_grad.T * G_matrix
    a = V_grad.T * f_vector
    b_norm_squared = b_T * R_matrix.inv() * b_T.T
    x_norm_squared = Matrix([[x1, x2]]) * Q_matrix * Matrix([[x1], [x2]])
    
    #lambda_x = (sqrt(a[0] ** 2 + x_norm_squared[0] * b_norm_squared[0]) + a[0]) / b_norm_squared[0]
    '''
    lambda_x = safe_divideSR(
        sqrt(a[0] ** 2 + x_norm_squared[0] * b_norm_squared[0]) + a[0],
        b_norm_squared[0],
        default=0
    )
    '''
    lambda_x = sqrt(a[0] ** 2 + x_norm_squared[0] * b_norm_squared[0]) + a[0] / b_norm_squared[0]
    
    lambda_x_diff = lambda_x.diff(x1) + lambda_x.diff(x2)
    
    #u_value = -R_matrix.inv() * b_T.T * lambda_x
    u_value = Matrix([0, u_func])
    V_dot = V_grad.T * f_vector + V_grad.T * G_matrix * u_value
    u_mag = u_value.T * R_matrix * u_value
    #return lambdify((x1, x2), V_value, 'numpy'), lambdify((x1, x2), V_dot[0], 'numpy'), lambdify((x1, x2), lambda_x, 'numpy'), lambdify((x1, x2), u_value[1], 'numpy'), lambdify((x1, x2), u_mag, 'numpy'), lambdify((x1, x2), lambda_x_diff, 'numpy')
    return V_value, V_dot[0], lambda_x, u_value[1], u_mag


