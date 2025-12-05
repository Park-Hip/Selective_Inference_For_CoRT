import numpy as np
from sklearn.linear_model import LassoCV, Lasso
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import pandas as pd
import math

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

def get_active_X(model_coef, X):
  active_cols = [idx for idx, coef in enumerate(model_coef) if coef != 0]
  inactive_cols = [idx for idx, coef in enumerate(model_coef) if coef == 0]

  return X[:, active_cols], X[:, inactive_cols]

def construct_test_statistic(y, j, X_active):
  '''
  Compute test statistic direction etaj and its projection etajTy
  '''
  n, p = X_active.shape
  ej = np.zeros((p,1))
  ej[j] = 1
  etajT = ej.T @ np.linalg.pinv(X_active.T @ X_active) @ X_active.T
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


def compute_lasso_interval(X, a, b, alpha_val, z_obs):
  """
  Returns the stability interval [L, R] for a single Lasso model
  containing the point z_obs.
  """
  a = a.reshape(-1, 1)
  b = b.reshape(-1, 1)

  # 1. Fit Lasso at z_obs to get the "Truth" (Active Set & Signs)
  n, p =  X.shape
  y = a + b * z_obs
  clf = Lasso(alpha=alpha_val, fit_intercept=False, max_iter=100000, tol=1e-12)
  clf.fit(X, y.flatten())

  # 2. Extract Active Set (M) and Signs (s)
  active_indices = [idx for idx, coef in enumerate(clf.coef_) if coef != 0]
  inactive_indices = [idx for idx, coef in enumerate(clf.coef_) if coef == 0]
  m = len(active_indices)

  # Global bounds for this model (initially infinite)
  L_model = -np.inf
  R_model = np.inf

  # if m > 0:
  #   resid = y.flatten() - X @ clf.coef_
  #   # Calculate correlation on ACTIVE features (should be equal to lambda)
  #   correlations = np.abs(X[:, active_indices].T @ resid)
  #   lambda_val = np.mean(correlations)
  # else:
  #   # If null model, lambda is effectively the active constraints boundary
  #   # But for the inequalities, we use the theoretical max lambda from input
  #   # Standard relation: lambda_math = n * alpha_sklearn
  #   lambda_val = n * alpha_val

  lambda_val = n * alpha_val

  # # Safety check for numerical zero (prevent divide by zero)
  # if lambda_val < 1e-10:
  #   lambda_val = 1e-10

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

def get_Z_train(z_obs, folds, source_data, a_global, b_global, lamda, K, T):
  """
  Computes the Z_train interval containing z_obs.
  Intersection of stability regions for T Baselines and T*K Augmented models.
  """
  L_final = - np.inf
  R_final = np.inf

  fold_indices = []
  start = 0
  for f in folds:
    size = f["X"].shape[0]
    fold_indices.append(np.arange(start, start+size))
    start += size

  for t in range(T):
    train_indices_list = [fold_indices[i] for i in range(T) if i != t]
    train_indices = np.concatenate(train_indices_list)

    X_train_list = [folds[i]["X"] for i in range(T) if i != t]
    X_target_train = np.vstack(X_train_list)

    X_base, a_base, b_base = get_affine_params(X_target_train, train_indices, a_global, b_global)
    L_base, R_base = compute_lasso_interval(X_base, a_base, b_base, lamda, z_obs)

    L_final = max(L_final, L_base)
    R_final = min(R_final, R_base)

    for k in range(K):
      source_data_k = source_data[k]

      X_aug, a_aug, b_aug = get_affine_params(X_target_train, train_indices, a_global, b_global, source_data_k)
      L_aug, R_aug = compute_lasso_interval(X_aug, a_aug, b_aug, lamda, z_obs)

      L_final = max(L_final, L_aug)
      R_final = min(R_final, R_aug)

  return L_final, R_final

def get_u_v(X, a, b, z_obs, alpha_val):
  n, p = X.shape
  a = a.reshape(-1, 1)
  b = b.reshape(-1, 1)

  y = a + b * z_obs
  clf = Lasso(alpha=alpha_val, fit_intercept=False, tol=0.0001, max_iter=10000)
  clf.fit(X, y.flatten())

  active_indices = [idx for idx, coef in enumerate(clf.coef_) if coef != 0]
  inactive_indices = [idx for idx, coef in enumerate(clf.coef_) if coef == 0]
  m = len(active_indices)

  u_full = np.zeros((p, 1))
  v_full = np.zeros((p, 1))

  X_M = X[:, active_indices]
  X_Mc = X[:, inactive_indices]
  s_M = np.sign(clf.coef_[active_indices]).reshape(-1, 1)

  lambda_val = alpha_val * n

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

def solve_quadratic_interval(A, B, C, z_obs):
  roots = np.roots([A, B, C])
  real_roots = sorted([r.real for r in roots if np.isreal(r)])

  L = -np.inf
  R = np.inf

  for r in real_roots:
    if r < z_obs:
      L = max(L, r)
    elif r > z_obs:
      R = min(R, r)

  return L, R

def get_Z_val(folds, T, K, a_global, b_global, z_obs, alpha_val, source_data):
  L_final = - np.inf
  R_final = np.inf

  fold_indices = []
  start = 0
  for f in folds:
    size = f["X"].shape[0]
    fold_indices.append(np.arange(start, start + size))
    start += size

  for t in range(T):
    X_train_list = [folds[i]["X"] for i in range(T) if i != t]
    X_target_train = np.vstack(X_train_list)
    train_indices = [fold_indices[i] for i in range(T) if i!= t]
    X_base_train, a_base_train, b_base_train = get_affine_params(X_target_train, train_indices, a_global, b_global)
    u_base, v_base = get_u_v(X_base_train, a_base_train, b_base_train, z_obs, alpha_val)

    X_val = folds[t]["X"]
    val_indices = fold_indices[t]
    _, a_base_val, b_base_val = get_affine_params(X_val, val_indices, a_global, b_global)
    C2_base, C1_base, C0_base = get_loss_coefs(a_base_val, b_base_val,  u_base, v_base, X_val)

    for k in range(K):
      source_data_k = source_data[k]
      X_aug_train, a_aug_train, b_aug_train = get_affine_params(X_target_train, train_indices, a_global, b_global, source_data_k)
      u_aug, v_aug = get_u_v(X_aug_train, a_aug_train, b_aug_train, z_obs, alpha_val)
      C2_aug, C1_aug, C0_aug = get_loss_coefs(a_base_val, b_base_val, u_aug, v_aug, X_val)

      A_dif = C2_aug - C2_base
      B_dif = C1_aug - C1_base
      C_dif = C0_aug - C0_base

      L_vote, R_vote = solve_quadratic_interval(A_dif, B_dif, C_dif, z_obs)
      L_final = max(L_final, L_vote)
      R_final = min(R_final, R_vote)

  return L_final, R_final
































