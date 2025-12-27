import numpy as np
import utils
from sklearn.linear_model import Lasso

def compute_lasso_interval(X, a, b, alpha, z):
    """
    Returns the stability interval [L, R] for a single Lasso model
    containing the point z.
    """

    a = a.reshape(-1, 1)
    b = b.reshape(-1, 1)

    # 1. Fit Lasso at z
    n, p =  X.shape
    y = a + b * z
    clf = Lasso(alpha=alpha, fit_intercept=False, tol=1e-10, max_iter=10000000)
    clf.fit(X, y.flatten())

    # 2. Extract Active Set (M) and Signs (s)
    active_indices = [idx for idx, coef in enumerate(clf.coef_) if coef != 0]
    inactive_indices = [idx for idx, coef in enumerate(clf.coef_) if coef == 0]
    m = len(active_indices)

    L_model = -np.inf
    R_model = np.inf

    # Standard relation: lambda_math = n * alpha_sklearn

    lambda_val = n * alpha

    if m == 0:
      # If no features selected, we only check inactive stationarity: |X'y| <= lambda
      # p = (1/lambda) X'a
      # q = (1/lambda) X'b
      p = (1 / lambda_val) * X.T @ a
      q = (1 / lambda_val) * X.T @ b

      # Constraints: -1 <= p + qz <= 1
      # 1. qz <= 1 - p
      # 2. -qz <= 1 + p
      A = np.concatenate([q.flatten(), -q.flatten()])
      B = np.concatenate([(1 - p).flatten(), (1 + p).flatten()])

    else:
      X_M = X[:, active_indices]
      X_Mc = X[:, inactive_indices]
      s_M = np.sign(clf.coef_[active_indices]).reshape(-1,1)

      P_M = X_M @ np.linalg.pinv(X_M.T @ X_M) @ X_M.T
      u = np.linalg.pinv(X_M.T @ X_M) @ (X_M.T @ a - lambda_val * s_M)
      v = np.linalg.pinv(X_M.T @ X_M) @ (X_M.T @ b)
      p = (1/lambda_val) * X_Mc.T @ (np.eye(n) - P_M) @ a + X_Mc.T @ X_M @ np.linalg.pinv(X_M.T @ X_M) @ s_M
      q = (1/lambda_val) * X_Mc.T @ (np.eye(n) - P_M) @ b

      # 4. Construct Inequalities (Psi * z <= Gamma)
      # Constraint 1: Sign Consistency (-diag(s)*v*z < diag(s)*u)
      # A1 * z <= B1
      A1 = - np.diag(s_M.flatten()) @ v
      B1 = np.diag(s_M.flatten()) @ u

      # Constraint 2: Inactive Stationarity (|s_Mc| < 1)
      # q*z <= 1-p  AND  -q*z <= 1+p
      ones = np.ones((len(inactive_indices), 1))

      # q*z <= 1 - p
      A2 = q
      B2 = ones - p

      # -q*z <= 1 + p
      A3 = -q
      B3 = ones + p

      A = np.concatenate([A1.flatten(), A2.flatten(), A3.flatten()])
      B = np.concatenate([B1.flatten(), B2.flatten(), B3.flatten()])

    # 5. Solve for Interval [L, R]
    # For each inequality A_i * z <= B_i:
    # If A_i > 0: z <= B_i / A_i  (Upper Bound)
    # If A_i < 0: z >= B_i / A_i  (Lower Bound)

    pos_idx = A > 1e-9
    if np.any(pos_idx):
      upper_bound = B[pos_idx] / A [pos_idx]
      R_model = np.min(upper_bound)

    neg_idx = A < -1e-9
    if np.any(neg_idx):
      lower_bound = B[neg_idx] / A[neg_idx]
      L_model = np.max(lower_bound)

    return L_model, R_model

def get_u_v(X, a, b, z, alpha):
    n, p = X.shape
    a = a.reshape(-1, 1)
    b = b.reshape(-1, 1)

    y = a + b * z
    clf = Lasso(alpha=alpha, fit_intercept=False, tol=1e-10, max_iter=10000000)
    clf.fit(X, y.flatten())

    active_indices = [idx for idx, coef in enumerate(clf.coef_) if coef != 0]
    inactive_indices = [idx for idx, coef in enumerate(clf.coef_) if coef == 0]
    m = len(active_indices)

    u_full = np.zeros((p, 1))
    v_full = np.zeros((p, 1))

    if m > 0:
      X_M = X[:, active_indices]
      X_Mc = X[:, inactive_indices]
      s_M = np.sign(clf.coef_[active_indices]).reshape(-1, 1)

      lambda_val = alpha * n

      u_active = np.linalg.pinv(X_M.T @ X_M) @ (X_M.T @ a - lambda_val * s_M)
      v_active = np.linalg.pinv(X_M.T @ X_M) @ (X_M.T @ b)

      u_full[active_indices] = u_active
      v_full[active_indices] = v_active

    return u_full, v_full

def get_loss_coefs(a_val, b_val, u, v, X_val):
    """
    Calculates coefficients for ||psi + omega*z||^2
    """

    a_val = a_val.reshape(-1, 1)
    b_val = b_val.reshape(-1, 1)

    phi = a_val - X_val @ u
    omega = b_val - X_val @ v

    C2 = (omega.T @ omega).item()
    C1 = 2 * (phi.T @ omega).item()
    C0 = (phi.T @ phi).item()

    return C2, C1, C0

def solve_quadratic_interval(A, B, C, z):
    roots = np.roots([A, B, C])
    real_roots = sorted([r.real for r in roots if np.isreal(r)])

    L = -np.inf
    R = np.inf

    for r in real_roots:
      if r < z:
        L = max(L, r)
      elif r > z:
        R = min(R, r)

    return L, R

def find_interval(t, k, z_min, z_max, folds, source_data, a_global, b_global, lamda, K, T): 
    fold_indices = []
    start = 0
    for f in folds:
      size = f["X"].shape[0]
      fold_indices.append(np.arange(start, start+size))
      start += size

    train_indices_list = []
    X_train_list = []
    for i in range(T):
        if i != t:
            train_indices_list.append(fold_indices[i])
            X_train_list.append(folds[i]["X"])

    train_indices = np.concatenate(train_indices_list)
    X_target_train = np.vstack(X_train_list)
    X_base, a_base, b_base = utils.get_affine_params(X_target_train, train_indices, a_global, b_global)

    z1 = z_min
    lim1 = z_max
    interval = []
    while z1 < lim1:
        L_base, R_base = compute_lasso_interval(X_base, a_base, b_base, lamda, z1)
        source_data_k = source_data[k]
        X_aug, a_aug, b_aug = utils.get_affine_params(X_target_train, train_indices, a_global, b_global, source_data_k)
        
        z2 = z1
        lim2 = min(R_base, lim1)
        # print(z2, lim2)
        while z2 < lim2:
            L_aug, R_aug = compute_lasso_interval(X_aug, a_aug, b_aug, lamda, z2)
            z3 = z2
            lim3 = min(R_aug, lim2)
            # print(z3, lim3)
            # print(f"Starting wih left: {z3}, right: {lim3}"
            while z3 < lim3:
                u_base, v_base = get_u_v(X_base, a_base, b_base, z3, lamda)
                # print(z3)
                X_val = folds[t]["X"]
                val_indices = fold_indices[t]
                
                _, a_base_val, b_base_val = utils.get_affine_params(X_val, val_indices, a_global, b_global)
                C2_base, C1_base, C0_base = get_loss_coefs(a_base_val, b_base_val, u_base, v_base, X_val)

                u_aug, v_aug = get_u_v(X_aug, a_aug, b_aug, z3, lamda)
                C2_aug, C1_aug, C0_aug = get_loss_coefs(a_base_val, b_base_val, u_aug, v_aug, X_val)

                A_dif = C2_aug - C2_base
                B_dif = C1_aug - C1_base
                C_dif = C0_aug - C0_base

                L_vote, R_vote = solve_quadratic_interval(A_dif, B_dif, C_dif, z3)
                delta = A_dif * z3 * z3 + B_dif * z3 + C_dif
                cnt = 0
                if delta <= 0: 
                    cnt = 1
                else:
                    cnt = 0
                l = z3
                r = min(R_vote, lim3)
                interval.append([l, r, cnt])
                z3 = max(z3, R_vote) + 1e-5 
            z2 = max(R_aug, z2) + 1e-5 
        z1 = max(z1, R_base) + 1e-5

    return interval


def find_similar_source(z, z_max, interval, K, T):
    similar_source = []
    ans = z_max
    for k in range(K):
        cnt = 0
        for t in range(T):
            for val in interval[k, t]:
                if val[0] <= z and z <= val[1]:
                    cnt += val[2]
                    ans = min(ans, val[1])
        
        if cnt >= (T + 1) / 2:
            similar_source.append(k)
    return ans, similar_source

def get_Z_CoRT(X_combined, similar_source_index, alpha, a_global, b_global, source_data, z):
    a_CoRT_list = []
    b_CoRT_list = []

    for k in similar_source_index:
      y_k = source_data[k]["y"].ravel()
      a_CoRT_list.append(y_k)
      b_CoRT_list.append(np.zeros(len(y_k)))

    a_CoRT_list.append(a_global.ravel())
    b_CoRT_list.append(b_global.ravel())

    a_CoRT = np.hstack(a_CoRT_list)
    b_CoRT = np.hstack(b_CoRT_list)

    a_CoRT = a_CoRT.reshape(-1,1)
    b_CoRT = b_CoRT.reshape(-1,1)

    y_combined = a_CoRT + b_CoRT * z

    n, p = X_combined.shape

    clf = Lasso(alpha=alpha, fit_intercept=False, tol=1e-10, max_iter=10000000)
    clf.fit(X_combined, y_combined.flatten())

    active_indices = [idx for idx, coef in enumerate(clf.coef_) if coef != 0]
    inactive_indices = [idx for idx, coef in enumerate(clf.coef_) if coef == 0]
    m = len(active_indices)

    L_CoRT = -np.inf
    R_CoRT = np.inf

    lambda_val = n * alpha

    if m == 0:
      # Inactive Constraints Only: |X'y| <= lambda
      p = (1 / lambda_val) * X_combined.T @ a_CoRT
      q = (1 / lambda_val) * X_combined.T @ b_CoRT

      A = np.concatenate([q.flatten(), -q.flatten()])
      B = np.concatenate([(1 - p).flatten(), (1 + p).flatten()])

    else:
      X_M = X_combined[:, active_indices]
      X_Mc = X_combined[:, inactive_indices]
      s_M = np.sign(clf.coef_[active_indices]).reshape(-1, 1)

      P_M = X_M @ np.linalg.pinv(X_M.T @ X_M) @ X_M.T
      u = np.linalg.pinv(X_M.T @ X_M) @ (X_M.T @ a_CoRT - lambda_val * s_M)
      v = np.linalg.pinv(X_M.T @ X_M) @ (X_M.T @ b_CoRT)
      p = (1 / lambda_val) * X_Mc.T @ (np.eye(n) - P_M) @ a_CoRT + X_Mc.T @ X_M @ np.linalg.pinv(X_M.T @ X_M) @ s_M
      q = (1 / lambda_val) * X_Mc.T @ (np.eye(n) - P_M) @ b_CoRT

      # 4. Construct Inequalities (Psi * z <= Gamma)

      # Constraint 1: Sign Consistency (-diag(s)*v*z < diag(s)*u)
      # A1 * z <= B1
      A1 = - np.diag(s_M.flatten()) @ v
      B1 = np.diag(s_M.flatten()) @ u

      # Constraint 2: Inactive Stationarity (|s_Mc| < 1)
      # q*z <= 1-p  AND -q*z <= 1+p
      ones = np.ones((len(inactive_indices), 1))

      # q*z <= 1 - p
      A2 = q
      B2 = ones - p

      # -q*z <= 1 + p
      A3 = -q
      B3 = ones + p

      A = np.concatenate([A1.flatten(), A2.flatten(), A3.flatten()])
      B = np.concatenate([B1.flatten(), B2.flatten(), B3.flatten()])

    # 5. Solve for Interval [L, R]
    # For each inequality A_i * z <= B_i:
    # If A_i > 0: z <= B_i / A_i  (Upper Bound)
    # If A_i < 0: z >= B_i / A_i  (Lower Bound)

    pos_idx = A > 1e-9
    if np.any(pos_idx):
      upper_bound = B[pos_idx] / A[pos_idx]
      R_CoRT = np.min(upper_bound)

    neg_idx = A < -1e-9
    if np.any(neg_idx):
      lower_bound = B[neg_idx] / A[neg_idx]
      L_CoRT = np.max(lower_bound)

    return L_CoRT, R_CoRT, active_indices

    

