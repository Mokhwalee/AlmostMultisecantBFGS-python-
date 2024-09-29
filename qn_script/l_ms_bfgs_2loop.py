import numpy as np


# Example usage:
# x0 = np.zeros((n, 1))
# stepsize = 0.01
# max_iter = 100
# L = 10
# fn = lambda x: ...  # Define function to minimize
# grad = lambda x: ...  # Define gradient function
# f_optimal, traj_opt, x_opt = l_ms_bfgs_2loop(x0, stepsize, max_iter, L, fn, grad)


def get_l_ms_bfgs_2loop(Sk, Yk, grad):
    q = grad
    alpha = []
    rho = []

    for i in range(len(Sk) - 1, -1, -1):
        s = Sk[i]
        y = Yk[i]
        rho_i = 1.0 / (y.T @ s)
        alpha_i = rho_i * (s.T @ q)
        q = q - alpha_i * y
        alpha.append(alpha_i)
        rho.append(rho_i)

    r = q  # Initial Hessian is identity

    for i in range(len(Sk)):
        s = Sk[i]
        y = Yk[i]
        beta = rho[i] * (y.T @ r)
        r = r + s * (alpha[i] - beta)

    return r

def l_ms_bfgs_2loop(x0, stepsize, max_iter, L, fn, grad):
    traj_opt = np.full(max_iter, np.inf)
    smem = []
    ymem = []
    x = x0
    x_opt = x0
    f_optimal = np.inf

    # initialize the lists for L-MS-BFGS
    Sk = []
    Yk = []

    for iter in range(max_iter):
        if iter == 0:
            xn = x - grad(x) * stepsize
        else:
            Bg = get_l_ms_bfgs_2loop(Sk, Yk, grad(x))
            xn = x - Bg * stepsize

        s = xn - x
        y = grad(xn) - grad(x)
        smem.append(s)
        ymem.append(y)

        # simplest multisecant
        if len(smem) > L:
            smem.pop(0)
            ymem.pop(0)

        x = xn
        traj_opt[iter] = fn(x)

        if fn(x) < f_optimal:
            x_opt = x
            f_optimal = traj_opt[iter]

        # Save Sk and Yk
        Sk.append(smem.copy())
        Yk.append(ymem.copy())

        if len(Sk) > L:
            Sk.pop(0)
            Yk.pop(0)

        # stopping criteria
        if fn(x) < 1e-14:
            traj_opt[iter + 1:max_iter] = fn(x)
            break

    return f_optimal, traj_opt, x_opt