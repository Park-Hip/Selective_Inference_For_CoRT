import numpy as np
import math
from mpmath import mp
from sklearn.linear_model import Lasso

def split_target(T, X_target, y_target, n_target):
  folds = []
  fold_size = math.floor(n_target / T)

  for i in range(T):
    start = i * fold_size
    if i == T-1:
      end = n_target
    else:
      end =  (i + 1) * fold_size

    X_fold = X_target[start:end]
    y_fold = y_target[start:end]

    folds.append({"X": X_fold, "y": y_fold})
  return folds

def get_u_v(X, a, b, z, alpha):
    n, p = X.shape
    a = a.reshape(-1, 1)
    b = b.reshape(-1, 1)

    y = a + b * z
    clf = Lasso(alpha=alpha, fit_intercept=False, tol=1e-10, max_iter=100000)
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
    a_val = a_val.reshape(-1, 1)
    b_val = b_val.reshape(-1, 1)

    phi = a_val - X_val @ u
    omega = b_val - X_val @ v

    C2 = (omega.T @ omega).item()
    C1 = 2 * (phi.T @ omega).item()
    C0 = (phi.T @ phi).item()

    return C2, C1, C0

def get_active_X(model_coef, X):
  active_cols = [idx for idx, coef in enumerate(model_coef) if coef != 0]
  inactive_cols = [idx for idx, coef in enumerate(model_coef) if coef == 0]

  return X[:, active_cols], X[:, inactive_cols]

def construct_test_statistic(y, j, X_active):
  n, m = X_active.shape
  ej = np.zeros((m,1))
  ej[j] = 1
  etajT = ej.T @ np.linalg.pinv(X_active.T @ X_active + 1e-6 * np.eye(m)) @ X_active.T 
  etaj = etajT.T
  etajTy = etajT @ y

  return etaj, etajTy.item()

def get_affine_params(X_fold, y_fold_indices, a_global, b_global, source_data_k=None):
  X_out = X_fold
  a_out = a_global[y_fold_indices].ravel()
  b_out = b_global[y_fold_indices].ravel()

  if source_data_k is not None:
    X_source = source_data_k["X"]
    y_source = source_data_k["y"].ravel()
    n_source = len(y_source)

    X_out = np.vstack([X_out, X_source])
    a_out = np.hstack([a_out, y_source])
    b_zero = np.zeros(n_source)
    b_out = np.hstack([b_out, b_zero])

  return X_out, a_out, b_out

def pivot(z_interval, etaj, etajTy, tn_mu, cov):
    new_z_interval = []
    for interval in z_interval:
        if len(new_z_interval) == 0:
            new_z_interval.append(interval)
        else:
            dif = abs(interval[0] - new_z_interval[-1][1])
            if dif < 0.0001:
                new_z_interval[-1][1] = interval[1]
            else:
                new_z_interval.append(interval)
    z_interval = new_z_interval

    tn_sigma = (np.sqrt(etaj.T @ cov @ etaj)).item()

    num = 0
    den = 0

    for interval in z_interval:
        lower = interval[0]
        upper = interval[1]
        z_u = (upper - tn_mu) / tn_sigma
        z_l = (lower - tn_mu) / tn_sigma
        den += mp.ncdf(z_u) - mp.ncdf(z_l)
        if etajTy >= upper:
            num += mp.ncdf(z_u) - mp.ncdf(z_l)

        elif lower <= etajTy < upper:
            z_norm = (etajTy - tn_mu) / tn_sigma
            num += mp.ncdf(z_norm) - mp.ncdf(z_l)

    if den == 0:
        return None

    conditional_cdf = num / den
    p_value = 2 * min(conditional_cdf, 1 - conditional_cdf)
    
    return float(p_value)

def combine_Z(L_train, R_train, L_val, R_val, L_CoRT, R_CoRT):
  L = [L_train, L_val, L_CoRT]
  R = [R_train, R_val, R_CoRT]

  L_final = max(L)
  R_final = min(R)

  return L_final, R_final

def computed_truncated_cdf(L, R, z, mu, sigma):
  norm_L = (L - mu) / sigma
  norm_R = (R - mu) / sigma
  norm_z = (z - mu) / sigma

  cdf_L = mp.ncdf(norm_L)
  cdf_R = mp.ncdf(norm_R)
  cdf_y = mp.ncdf(norm_z)

  denominator = cdf_R - cdf_L

  if denominator == 0:
      return None
  
  numerator = cdf_y - cdf_L

  if numerator < 0:
      print("numerator is negative")

  if numerator > denominator:
      print("numerator is bigger than denominator")
  val = numerator / denominator
  return float(val)
