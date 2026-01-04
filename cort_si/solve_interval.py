import numpy as np
from sklearn.linear_model import Lasso

def compute_similar_source(z, z_max, interval, K, T):
    similar_source = []
    L_final = -z_max
    R_final = z_max
    for k in range(K):
        C_k = 0
        for t in range(T):
            for [l, r, cnt] in interval[k, t]:
                # print(l, r, cnt)
                if l <= z and z <= r:
                    C_k += cnt
                    L_final = max(L_final, l)
                    R_final = min(R_final, r)
        
        if C_k >= (T + 1) / 2:
            similar_source.append(k)
    return L_final, R_final, similar_source

def compute_quadratic_interval(A, B, C, z):
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

def compute_lasso_interval(X, a, b, lamda, z, similar_source_index = None, source_data = None):
    if similar_source_index != None:
        a_list = []
        b_list = []
        for k in similar_source_index:
            y_k = source_data[k]["y"].ravel()
            a_list.append(y_k)
            b_list.append(np.zeros(len(y_k)))
        a_list.append(a.ravel())
        b_list.append(b.ravel())
        a = np.hstack(a_list)
        b = np.hstack(b_list)
    a = a.reshape(-1, 1)
    b = b.reshape(-1, 1)
    n, p =  X.shape
    y = a + b * z
    clf = Lasso(alpha=lamda, fit_intercept=False, tol=1e-10, max_iter=100000)
    clf.fit(X, y.flatten())
    active_indices = [idx for idx, coef in enumerate(clf.coef_) if coef != 0]
    inactive_indices = [idx for idx, coef in enumerate(clf.coef_) if coef == 0]
    m = len(active_indices)
    L_model = -np.inf
    R_model = np.inf
    lambda_val = n * lamda

    if m == 0:
      p = (1 / lambda_val) * X.T @ a
      q = (1 / lambda_val) * X.T @ b
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
      A1 = - np.diag(s_M.flatten()) @ v
      B1 = np.diag(s_M.flatten()) @ u
      ones = np.ones((len(inactive_indices), 1))
      A2 = q
      B2 = ones - p
      A3 = -q
      B3 = ones + p
      A = np.concatenate([A1.flatten(), A2.flatten(), A3.flatten()])
      B = np.concatenate([B1.flatten(), B2.flatten(), B3.flatten()])
    pos_idx = A > 1e-9
    if np.any(pos_idx):
      upper_bound = B[pos_idx] / A [pos_idx]
      R_model = np.min(upper_bound)
    neg_idx = A < -1e-9
    if np.any(neg_idx):
      lower_bound = B[neg_idx] / A[neg_idx]
      L_model = np.max(lower_bound)

    if similar_source_index == None:
       return L_model, R_model
    return L_model, R_model, active_indices
