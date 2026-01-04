import numpy as np
from . import solve_interval, utils
from sklearn.linear_model import Lasso

def get_Z_base_aug(z_obs, folds, source_data, a_global, b_global, lamda_not_source, lamda_1_source, K, T):
  list_R = []
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

    X_base, a_base, b_base = utils.get_affine_params(X_target_train, train_indices, a_global, b_global)
    L_base, R_base = solve_interval.compute_lasso_interval(X_base, a_base, b_base, lamda_not_source, z_obs)

    L_final = max(L_final, L_base)
    R_final = min(R_final, R_base)

    for k in range(K):
      source_data_k = source_data[k]

      X_aug, a_aug, b_aug = utils.get_affine_params(X_target_train, train_indices, a_global, b_global, source_data_k)
      L_aug, R_aug = solve_interval.compute_lasso_interval(X_aug, a_aug, b_aug, lamda_1_source, z_obs)

      L_final = max(L_final, L_aug)
      R_final = min(R_final, R_aug)

  return L_final, R_final

def get_Z_val(folds, T, K, a_global, b_global, z_obs, lamda_not_source, lamda_1_source, source_data):
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

    train_indices_list = [fold_indices[i] for i in range(T) if i != t] ##
    train_indices = np.concatenate(train_indices_list) ##

    X_base_train, a_base_train, b_base_train = utils.get_affine_params(X_target_train, train_indices, a_global, b_global)
    u_base, v_base = utils.get_u_v(X_base_train, a_base_train, b_base_train, z_obs, lamda_not_source)

    X_val = folds[t]["X"]
    val_indices = fold_indices[t]
    _, a_base_val, b_base_val = utils.get_affine_params(X_val, val_indices, a_global, b_global)
    C2_base, C1_base, C0_base = utils.get_loss_coefs(a_base_val, b_base_val,  u_base, v_base, X_val)

    for k in range(K):
      source_data_k = source_data[k]
      X_aug_train, a_aug_train, b_aug_train = utils.get_affine_params(X_target_train, train_indices, a_global, b_global, source_data_k)
      u_aug, v_aug = utils.get_u_v(X_aug_train, a_aug_train, b_aug_train, z_obs, lamda_1_source)
      C2_aug, C1_aug, C0_aug = utils.get_loss_coefs(a_base_val, b_base_val, u_aug, v_aug, X_val)

      A_dif = C2_aug - C2_base
      B_dif = C1_aug - C1_base
      C_dif = C0_aug - C0_base

      L_vote, R_vote = solve_interval.compute_quadratic_interval(A_dif, B_dif, C_dif, z_obs)
      L_final = max(L_final, L_vote)
      R_final = min(R_final, R_vote)

  return L_final, R_final

def get_Z_CoRT(X_combined, similar_source_index, lamda, a_global, b_global, source_data, z_obs):
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

  y_combined = a_CoRT + b_CoRT * z_obs

  n, p = X_combined.shape

  clf = Lasso(alpha=lamda, fit_intercept=False, tol=1e-10, max_iter=100000)
  clf.fit(X_combined, y_combined)

  active_indices = [idx for idx, coef in enumerate(clf.coef_) if coef != 0]
  inactive_indices = [idx for idx, coef in enumerate(clf.coef_) if coef == 0]
  m = len(active_indices)

  L_CoRT = -np.inf
  R_CoRT = np.inf

  lambda_val = n * lamda

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
    upper_bound = B[pos_idx] / A[pos_idx]
    R_CoRT = np.min(upper_bound)

  neg_idx = A < -1e-9
  if np.any(neg_idx):
    lower_bound = B[neg_idx] / A[neg_idx]
    L_CoRT = np.max(lower_bound)
 
  return L_CoRT, R_CoRT, active_indices