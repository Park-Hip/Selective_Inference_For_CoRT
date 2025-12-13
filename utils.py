import numpy as np
from sklearn.linear_model import Lasso
import math
from scipy.stats import norm
from joblib import Parallel, delayed

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
  n, m = X_active.shape
  ej = np.zeros((m,1))
  ej[j] = 1
  etajT = ej.T @ np.linalg.pinv(X_active.T @ X_active + 1e-6 * np.eye(m)) @ X_active.T #####
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
  clf = Lasso(alpha=alpha_val, fit_intercept=False, tol=1e-8, max_iter=1000000  )
  clf.fit(X, y.flatten())

  # 2. Extract Active Set (M) and Signs (s)
  active_indices = [idx for idx, coef in enumerate(clf.coef_) if coef != 0]
  inactive_indices = [idx for idx, coef in enumerate(clf.coef_) if coef == 0]
  m = len(active_indices)

  # Global bounds for this model (initially infinite)
  L_model = -np.inf
  R_model = np.inf

  #   # Standard relation: lambda_math = n * alpha_sklearn
  #   lambda_val = n * alpha_val

  lambda_val = n * alpha_val

  if m == 0:
    # If no features selected, we only check inactive stationarity: |X'y| <= lambda
    # p = (1/lambda) X'a
    # q = (1/lambda) X'b
    p= (1 / lambda_val) * X.T @ a
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

def get_Z_train(z_obs, folds, source_data, a_global, b_global, lamda, K, T):
  """
  Computes the Z_train interval containing z_obs.
  Intersection of stability regions for T Baselines and T*K Augmented models.
  """

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
  clf = Lasso(alpha=alpha_val, fit_intercept=False, tol=1e-8, max_iter=1000000)
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

    train_indices_list = [fold_indices[i] for i in range(T) if i != t] ##
    train_indices = np.concatenate(train_indices_list) ##

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

def get_Z_CoRT(X_combined, similar_source_index, alpha_val, a_global, b_global, source_data, z_obs):
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

  clf = Lasso(alpha=alpha_val, fit_intercept=False, tol=1e-8, max_iter=1000000)
  clf.fit(X_combined, y_combined)

  active_indices = [idx for idx, coef in enumerate(clf.coef_) if coef != 0]
  inactive_indices = [idx for idx, coef in enumerate(clf.coef_) if coef == 0]
  m = len(active_indices)

  L_CoRT = -np.inf
  R_CoRT = np.inf

  lambda_val = n * alpha_val

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
    upper_bound = B[pos_idx] / A[pos_idx]
    R_CoRT = np.min(upper_bound)

  neg_idx = A < -1e-9
  if np.any(neg_idx):
    lower_bound = B[neg_idx] / A[neg_idx]
    L_CoRT = np.max(lower_bound)

  return L_CoRT, R_CoRT

def combine_Z(L_train, R_train, L_val, R_val, L_CoRT, R_CoRT):
  L = [L_train, L_val, L_CoRT]
  R = [R_train, R_val, R_CoRT]

  L_final = max(L)
  R_final = min(R)

  return L_final, R_final

def computed_truncated_cdf(L, R, z_obs, mu, sigma):
  normalized_L = (L-mu)/sigma
  normalized_R = (R-mu)/sigma
  normalized_z_obs = (z_obs-mu)/sigma

  truncated_cdf = (norm.cdf(normalized_z_obs, loc=0, scale=1) - norm.cdf(normalized_L, loc=0, scale=1))/(norm.cdf(normalized_R, loc=0, scale=1) - norm.cdf(normalized_L, loc=0, scale=1))

  return truncated_cdf

import numpy as np
from sklearn.linear_model import Lasso
import math
from scipy.stats import norm

class LassoPathTracker:
    def __init__(self, X, a, b, alpha, z_start):
        """
        Tracks the solution path of a SINGLE Lasso model beta(z).
        """
        self.X = X
        self.a = a.reshape(-1, 1)
        self.b = b.reshape(-1, 1)
        self.n, self.p_dim = X.shape
        self.lambda_val = self.n * alpha
        self.z_curr = z_start
        
        # 1. Initial Fit at z_start using Sklearn (Warm Start)
        y_start = self.a + self.b * z_start
        # High precision required for initial state
        clf = Lasso(alpha=alpha, fit_intercept=False, tol=1e-10, max_iter=100000)
        clf.fit(X, y_start.flatten())
        
        # 2. Store KKT State
        self.active = np.flatnonzero(clf.coef_)
        self.inactive = np.flatnonzero(clf.coef_ == 0)
        self.s = np.sign(clf.coef_[self.active]).reshape(-1, 1)
        
        # Pre-compute Gram inverse for active set
        self._update_matrices()
        
    def _update_matrices(self):
        """Re-computes u, v and Gram inverse based on current active set"""
        if len(self.active) == 0:
            self.u = np.zeros((self.p_dim, 1))
            self.v = np.zeros((self.p_dim, 1))
            self.Gram_inv = np.empty((0,0))
            return

        X_M = self.X[:, self.active]
        # Inverting small matrix (e.g. 10x10) is very fast
        self.Gram_inv = np.linalg.pinv(X_M.T @ X_M)
        
        # u = (X'X)^-1 (X'a - lambda*s)
        # v = (X'X)^-1 (X'b)
        self.u_M = self.Gram_inv @ (X_M.T @ self.a - self.lambda_val * self.s)
        self.v_M = self.Gram_inv @ (X_M.T @ self.b)
        
        # Map back to full size
        self.u = np.zeros((self.p_dim, 1))
        self.v = np.zeros((self.p_dim, 1))
        self.u[self.active] = self.u_M
        self.v[self.active] = self.v_M

    def get_next_knot(self):
        """
        Analytically finds the next z where the active set changes.
        Returns: (z_knot, event_type, index)
        """
        candidates = []
        
        # --- Event A: Variable LEAVING active set ---
        # Condition: beta_j(z) = u_j + v_j * z = 0
        # z = -u_j / v_j
        if len(self.active) > 0:
            # Avoid division by zero
            valid_v = np.abs(self.v_M) > 1e-9
            
            z_leave = -self.u_M[valid_v] / self.v_M[valid_v]
            
            # We only care about z > z_curr
            future_indices = np.where(z_leave > self.z_curr + 1e-6)[0]
            
            if len(future_indices) > 0:
                # Map back to real indices
                local_idx = np.where(valid_v)[0]
                best_local = local_idx[future_indices[np.argmin(z_leave[future_indices])]]
                z_min = z_leave[best_local].item()
                real_idx = self.active[best_local]
                candidates.append((z_min, 'leave', real_idx))

        # --- Event B: Variable ENTERING active set ---
        # Condition: |Correlation_j(z)| = lambda
        # Corr_j(z) = X_j' (y - X beta) = (X_j'a - X_j'X u) + z * (X_j'b - X_j'X v)
        #           = A_j + B_j * z
        
        # Residual params
        R_a = self.a - self.X @ self.u
        R_b = self.b - self.X @ self.v
        
        # Compute A and B for ALL inactive variables at once
        if len(self.inactive) > 0:
            X_Mc = self.X[:, self.inactive]
            A_vec = X_Mc.T @ R_a
            B_vec = X_Mc.T @ R_b
            
            # Solve 1: A + Bz = lambda  => z = (lambda - A) / B
            # Solve 2: A + Bz = -lambda => z = (-lambda - A) / B
            
            for i, idx in enumerate(self.inactive):
                B_val = B_vec[i].item()
                A_val = A_vec[i].item()
                
                if abs(B_val) < 1e-9: continue
                
                z1 = (self.lambda_val - A_val) / B_val
                z2 = (-self.lambda_val - A_val) / B_val
                
                if z1 > self.z_curr + 1e-6: candidates.append((z1, 'enter', idx))
                if z2 > self.z_curr + 1e-6: candidates.append((z2, 'enter', idx))

        if not candidates:
            return float('inf'), None, None
            
        # Return the nearest future event
        return min(candidates, key=lambda x: x[0])

    def update_state(self, z_knot, event_type, idx):
        """Updates internal state at the knot"""
        self.z_curr = z_knot
        
        if event_type == 'leave':
            # Remove idx from active
            mask = self.active != idx
            self.active = self.active[mask]
            self.s = self.s[mask]
            # Add to inactive
            self.inactive = np.sort(np.append(self.inactive, idx))
            
        elif event_type == 'enter':
            # Add idx to active
            self.active = np.sort(np.append(self.active, idx))
            # Remove from inactive
            self.inactive = self.inactive[self.inactive != idx]
            
            # Determine Sign (Continuity)
            # Correlation at knot determines sign. 
            # If Corr = +lambda, sign is +1.
            y_knot = self.a + self.b * z_knot
            # Need approximate beta to calculate correlation? 
            # Actually easier: check A + B*z sign
            # But we can just refit active set logic or calculate correlation directly
            # Fast check:
            corr = self.X[:, idx].T @ (y_knot - self.X @ self.u - self.X @ self.v * z_knot)
            new_sign = np.sign(corr).item()
            if new_sign == 0: new_sign = 1.0 # Fallback
            
            # Insert sign in correct position to match sorted active set
            pos = np.searchsorted(self.active, idx)
            # Rebuild s vector
            s_new = np.zeros((len(self.active), 1))
            # ... (Logic to insert sign is messy with numpy arrays, simpler to rebuild)
            # Let's just append and re-sort s based on active
            # Actually, reusing the correlation sign is safest.
            
            # Re-calculate ALL signs to be safe (prevent drift)
            # In a strict path algo we track signs, but recalculating 's' from correlation is robust
            pass 
            
            # Simplified update:
            # We must re-invert matrix anyway, let's just use the sign logic in _update_matrices
            # Wait, _update_matrices USES s. We need s first.
            
            # Insert s
            insert_pos = np.where(self.active == idx)[0][0]
            self.s = np.insert(self.s, insert_pos, new_sign).reshape(-1, 1)

        self._update_matrices()

# --- Helper Functions (Reused) ---

def split_target(T, X_target, y_target, n_target):
    # (Same as your previous kfold implementation)
    from sklearn.model_selection import KFold
    kf = KFold(n_splits=T, shuffle=True, random_state=42)
    folds = []
    for _, val_idx in kf.split(X_target):
        folds.append({"X": X_target[val_idx], "y": y_target[val_idx]})
    return folds

def get_loss_coefs(a_val, b_val, u, v, X_val):
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
    L, R = -np.inf, np.inf
    for r in real_roots:
        if r < z_obs: L = max(L, r)
        elif r > z_obs: R = min(R, r)
    return L, R

def combine_Z(L_train, R_train, L_val, R_val, L_CoRT, R_CoRT):
    return max([L_train, L_val, L_CoRT]), min([R_train, R_val, R_CoRT])

def precompute_matrices_for_tracker(folds, source_data, a_global, b_global, K, T):
    """Prepares the (X, a, b) matrices for all trackers"""
    configs = []
    
    # Helper
    def get_mats(X_in, idx, src=None):
        a_sub = a_global[idx].ravel()
        b_sub = b_global[idx].ravel()
        if src:
            X_out = np.vstack([X_in, src["X"]])
            a_out = np.hstack([a_sub, src["y"].ravel()])
            b_out = np.hstack([b_sub, np.zeros(len(src["y"]))])
        else:
            X_out = X_in
            a_out = a_sub
            b_out = b_sub
        return X_out, a_out, b_out

    # Indices logic (same as before)
    fold_indices = []
    start = 0
    for f in folds:
        size = f["X"].shape[0]
        fold_indices.append(np.arange(start, start+size))
        start += size

    for t in range(T):
        train_idx = np.concatenate([fold_indices[i] for i in range(T) if i != t])
        X_train_fold = np.vstack([folds[i]["X"] for i in range(T) if i != t])
        
        # Validation Config (for checking later)
        val_idx = fold_indices[t]
        _, a_val, b_val = get_mats(folds[t]["X"], val_idx)
        val_config = {"X": folds[t]["X"], "a": a_val, "b": b_val}

        # Baseline
        X, a, b = get_mats(X_train_fold, train_idx)
        configs.append({"type": "base", "t": t, "k": -1, "X": X, "a": a, "b": b, "val": val_config})
        
        # Augmented
        for k in range(K):
            X, a, b = get_mats(X_train_fold, train_idx, source_data[k])
            configs.append({"type": "aug", "t": t, "k": k, "X": X, "a": a, "b": b, "val": val_config})
            
    return configs


























