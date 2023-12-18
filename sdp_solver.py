import numpy as np


def is_PSD(mat):
    try:
        _ = np.linalg.cholesky(mat)
        return True
    except:
        return False


def formulate(rho, epsilon):
    # dimension of the target state
    dim = len(rho)
    # construct the constant matrix C
    rho_re = np.real(rho)
    rho_im = np.imag(rho)
    C = np.zeros([6 * dim + 3, 6 * dim + 3])
    C[:dim, :dim] = rho_re
    C[:dim , 2 * dim: 3 * dim] = -rho_im
    C[2 * dim: 3 * dim, :dim] = rho_im
    C[2 * dim: 3 * dim, 2 * dim: 3 * dim] = rho_re
    C[6 * dim, 6 * dim] = -1 + epsilon
    C[6 * dim + 1, 6 * dim + 1] = -1
    C[6 * dim + 2, 6 * dim + 2] = 1
    C = -C

    # construct the symmetric matrix associated with each variable
    num_var = 3 * (dim ** 2) + dim
    symmat_list = np.zeros([num_var, 6 * dim + 3, 6 * dim + 3])
    # associated with r_i
    for i in range(dim):
        symmat_list[i, 4 * dim + i, 4 * dim + i] = 1
        symmat_list[i, 5 * dim + i, 5 * dim + i] = 1
    # associated with q_ij
    offset = dim
    index = 0
    for i in range(dim):
        for j in range(i, dim):
            symmat_list[offset + index, dim + i, dim + j] = 1
            symmat_list[offset + index, dim + j, dim + i] = 1
            symmat_list[offset + index, 3 * dim + i, 3 * dim + j] = 1
            symmat_list[offset + index, 3 * dim + j, 3 * dim + i] = 1
            symmat_list[offset + index, 4 * dim + i, 4 * dim + j] = -1
            symmat_list[offset + index, 4 * dim + j, 4 * dim + i] = -1
            symmat_list[offset + index, 5 * dim + i, 5 * dim + j] = -1
            symmat_list[offset + index, 5 * dim + j, 5 * dim + i] = -1
            if i == j:
                symmat_list[offset + index, 6 * dim + 1, 6 * dim + 1] = 1
                symmat_list[offset + index, 6 * dim + 2, 6 * dim + 2] = -1
            index += 1
    # associated with tilde(q)_ij
    offset += int((dim + 1) * dim / 2)
    index = 0
    for i in range(dim):
        for j in range(i + 1, dim):
            symmat_list[offset + index, dim + i, 3 * dim + j] = -1
            symmat_list[offset + index, dim + j, 3 * dim + i] = 1
            symmat_list[offset + index, 3 * dim + i, dim + j] = 1
            symmat_list[offset + index, 3 * dim + j, dim + i] = -1
            symmat_list[offset + index, 4 * dim + i, 5 * dim + j] = 1
            symmat_list[offset + index, 4 * dim + j, 5 * dim + i] = -1
            symmat_list[offset + index, 5 * dim + i, 4 * dim + j] = -1
            symmat_list[offset + index, 5 * dim + j, 4 * dim + i] = 1
            index += 1
    # associated with x_ij
    offset += int((dim - 1) * dim / 2)
    index = 0
    for i in range(dim):
        for j in range(dim):
            symmat_list[offset + index, i, dim + j] = 1
            symmat_list[offset + index, dim + j, i] = 1
            symmat_list[offset + index, 2 * dim + i, 3 * dim + j] = 1
            symmat_list[offset + index, 3 * dim + j, 2 * dim + i] = 1
            if i == j:
                symmat_list[offset + index, 6 * dim, 6 * dim] = 1
            index += 1
    # associated with tilde(x)_ij
    offset += dim * dim
    index = 0
    for i in range(dim):
        for j in range(dim):
            symmat_list[offset + index, i, 3 * dim + j] = -1
            symmat_list[offset + index, dim + j, 2 * dim + i] = 1
            symmat_list[offset + index, 2 * dim + i, dim + j] = 1
            symmat_list[offset + index, 3 * dim + j, i] = -1
            index += 1

    # construct the vector in the objective
    b = np.zeros([1, num_var])
    b[0, :dim] = 1

    return symmat_list, C, b


def generate_search_direction(var_primal, mat_primal, var_dual, symmat_list, C, b, primal_list, dual_list, fea_tolerance=1e-3, gap_tolerance=1e-3, beta_bar=0.2, beta_star=0.1):
    num_var_primal = len(var_primal)
    dim_mat_primal = len(mat_primal)
    # check primal feasibility
    mat_primal_expected = np.zeros_like(C)
    for i in range(num_var_primal):
        mat_primal_expected += var_primal[i] * symmat_list[i]
    mat_primal_expected -= C
    P = mat_primal - mat_primal_expected
    max_diff_primal = np.max(np.abs(P))
    is_primal_feasible = True if max_diff_primal <= fea_tolerance else False
    # check dual feasibility
    d = b[0] - np.trace(symmat_list @ var_dual, axis1=1, axis2=2)
    max_diff_dual = np.max(np.abs(d))
    is_dual_feasible = True if max_diff_dual <= fea_tolerance else False
    is_feasible = False
    if is_primal_feasible and is_dual_feasible:
        is_feasible = True
        # compute duality gap
        value_primal = (b @ var_primal)[0,0]
        primal_list.append(value_primal)
        value_dual = np.trace(C @ var_dual)
        dual_list.append(value_dual)
        duality_gap = np.abs(value_primal - value_dual) / max(1, (np.abs(value_primal) + np.abs(value_dual)) / 2)
        if duality_gap <= gap_tolerance:
            return None, None, None, is_feasible, True
    else:
        primal_list.append((b @ var_primal)[0,0] if is_primal_feasible else np.nan)
        dual_list.append(np.trace(C @ var_dual) if is_dual_feasible else np.nan)

    # predictor procedure
    mat_primal_inv = np.linalg.inv(mat_primal)
    B = np.array([[np.trace(mat_primal_inv @ symmat_list[i] @ var_dual @ symmat_list[j]) for j in range(num_var_primal)] for i in range(num_var_primal)])
    XY = mat_primal @ var_dual
    R = -XY
    if not is_feasible:
        R += beta_bar * (np.trace(XY)) / dim_mat_primal * np.eye(dim_mat_primal)
    r = np.array([-d[i] + np.trace(symmat_list[i] @ (mat_primal_inv @ (R + P @ var_dual))) for i in range (num_var_primal)])
    # solve for dx_p
    dx_p, _, _, _ = np.linalg.lstsq(B, r, rcond=None)
    dx_p = dx_p.reshape([-1, 1])
    # compute dX_p
    dX_p = np.zeros_like(mat_primal)
    for i in range(num_var_primal):
        dX_p += dx_p[i] * symmat_list[i]
    dX_p -= P
    # compute dY_p
    dY_hat = mat_primal_inv @ (R - (dX_p @ var_dual))
    dY_p = (dY_hat + dY_hat.T) / 2

    # corrector procedure
    beta_aux = (np.trace((mat_primal + dX_p) @ (var_dual + dY_p)) / np.trace(XY)) ** 2
    if beta_aux > 1:
        beta_c = 1
    else:
        beta_c = max(beta_star, beta_aux) if is_feasible else max(beta_bar, beta_aux)
    # update R and r
    R = beta_c * (np.trace(XY)) / dim_mat_primal * np.eye(dim_mat_primal) - XY - (dX_p @ dY_p)
    r = np.array([-d[i] + np.trace(symmat_list[i] @ (mat_primal_inv @ (R + P @ var_dual))) for i in range (num_var_primal)])
    # solve for dx
    dx, _, _, _ = np.linalg.lstsq(B, r, rcond=None)
    dx = dx.reshape([-1, 1])
    # compute dX
    dX = np.zeros_like(mat_primal)
    for i in range(num_var_primal):
        dX += dx[i] * symmat_list[i]
    dX -= P
    # compute dY_p
    dY_hat = mat_primal_inv @ (R - (dX @ var_dual))
    dY = (dY_hat + dY_hat.T) / 2

    return dx, dX, dY, is_feasible, False


def compute_step_length(mat_primal, var_dual, dX, dY, is_feasible, a_max=100, gamma=0.9):
    a = a_max
    while not (is_PSD(mat_primal + a * dX) and is_PSD(var_dual + a * dY)):
        a *= 0.5
    a = gamma * a
    if not is_feasible:
        a = min(1, a)

    return a


def solve(rho, epsilon):
    symmat_list, C, b = formulate(rho, epsilon)
    var_primal = np.random.rand(len(symmat_list), 1)
    mat_primal = np.eye(len(C)) / len(C)
    var_dual = np.eye(len(C)) / len(C)
    primal_list = []
    dual_list = []
    stop_flag = False
    while not stop_flag:
        dx, dX, dY, is_feasible, stop_flag = generate_search_direction(var_primal, mat_primal, var_dual, symmat_list, C, b, primal_list, dual_list)
        if stop_flag:
            break
        a = compute_step_length(mat_primal, var_dual, dX, dY, is_feasible)
        var_primal += a * dx
        mat_primal += a * dX
        var_dual += a * dY

    return primal_list, dual_list


if __name__ == "__main__":
    # target state
    psi = np.sqrt([[0.3], [0.7]])
    rho = psi @ psi.T
    # 1 - targer root fidelity
    epsilon = 0
    # run the solver
    primal_list, dual_list = solve(rho, epsilon)
    print("The solved optimal primal value is", primal_list[-1])
    print("The solved optimal primal value is", dual_list[-1])
