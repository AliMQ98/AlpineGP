def compute_v_and_v_dot(V_value):
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

    lambda_x = safe_divide(
        sqrt(a[0] ** 2 + x_norm_squared[0] * b_norm_squared[0]) + a[0],
        b_norm_squared[0],
        default=0
    )
    lambda_x_diff = lambda_x.diff(x1) + lambda_x.diff(x2)
    u_value = -R_matrix.inv() * b_T.T * lambda_x
    V_dot = V_grad.T * f_vector + V_grad.T * G_matrix * u_value
    u_mag = u_value.T * R_matrix * u_value
    return lambdify((x1, x2), V_value, 'numpy'), lambdify((x1, x2), V_dot[0], 'numpy'), lambdify((x1, x2), lambda_x, 'numpy'), lambdify((x1, x2), u_value[1], 'numpy'), lambdify((x1, x2), u_mag, 'numpy'), lambdify((x1, x2), lambda_x_diff, 'numpy')

