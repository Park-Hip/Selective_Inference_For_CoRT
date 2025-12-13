#==========================##==========================#
import numpy as np
from sklearn.linear_model import Lasso
import math
from scipy.stats import norm
from joblib import Parallel, delayed


def precompute_parameters(folds, source_data, a_global, b_global, K, T):
    """
    Pre-calculates X, a, b for ALL models (Baseline and Augmented) once.
    Returns lists of dictionaries to be used in the loop.
    """
    train_models = [] # List of (X, a, b) for training stability
    val_configs = []  # List of (X_val, a_val, b_val) for validation voting

    # Helper to get params (renamed from get_affine_params for local scope)
    def get_params(X_fold, y_idx, src_k=None):
        X_out = X_fold
        a_out = a_global[y_idx].ravel()
        b_out = b_global[y_idx].ravel()
        
        if src_k is not None:
            # Stack Source Data
            X_out = np.vstack([X_out, src_k["X"]])
            a_out = np.hstack([a_out, src_k["y"].ravel()])
            b_out = np.hstack([b_out, np.zeros(len(src_k["y"]))]) # Source slope is 0
            
        return X_out, a_out.reshape(-1,1), b_out.reshape(-1,1)

    fold_indices = []
    start = 0
    for f in folds:
        size = f["X"].shape[0]
        fold_indices.append(np.arange(start, start+size))
        start += size

    for t in range(T):
        # 1. Prepare Training Indices
        # Concatenate indices for all folds EXCEPT t
        train_indices_list = [fold_indices[i] for i in range(T) if i != t]
        train_indices = np.concatenate(train_indices_list)
        
        X_train_list = [folds[i]["X"] for i in range(T) if i != t]
        X_target_train = np.vstack(X_train_list)

        # 2. Validation Params (Target Data only)
        val_idx = fold_indices[t]
        _, a_val, b_val = get_params(folds[t]["X"], val_idx)
        
        val_configs.append({
            "t": t, "X_val": folds[t]["X"], "a_val": a_val, "b_val": b_val
        })
        
        # 3. Baseline Model (Target Only)
        X_base, a_base, b_base = get_params(X_target_train, train_indices)
        
        train_models.append({
            "type": "base", "t": t, "k": -1,
            "X": X_base, "a": a_base, "b": b_base
        })
        
        # 4. Augmented Models (Target + Source k)
        for k in range(K):
            X_aug, a_aug, b_aug = get_params(X_target_train, train_indices, source_data[k])
            
            train_models.append({
                "type": "aug", "t": t, "k": k,
                "X": X_aug, "a": a_aug, "b": b_aug
            })

    return train_models, val_configs

def compute_lasso_interval_optimized(model_dict, alpha_val, z_obs):
    """
    Wrapper for parallel execution. Unpacks the dict and calls logic.
    """
    X = model_dict["X"]
    a = model_dict["a"]
    b = model_dict["b"]

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

      Gram_inv = np.linalg.pinv(X_M.T @ X_M)
      u = Gram_inv @ (X_M.T @ a - lambda_val * s_M)
      v = Gram_inv @ (X_M.T @ b)
      
      R_a = a - X_M @ u
      R_b = b - X_M @ v
      p = (1/lambda_val) * X_Mc.T @ R_a
      q = (1/lambda_val) * X_Mc.T @ R_b
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

# --- OPTIMIZED Z_TRAIN ---
def get_Z_train_fast(train_models, alpha_val, z_obs):
    # Run all Lasso models in parallel
    results = Parallel(n_jobs=-1, prefer="threads")(
        delayed(compute_lasso_interval_optimized)(m, alpha_val, z_obs) for m in train_models
    )
    
    # Aggregate
    L_final = max([res[0] for res in results])
    R_final = min([res[1] for res in results])
    return L_final, R_final

def get_Z_val_fast(train_models, val_configs, alpha_val, z_obs, K, T):
    
    def solve_single_val_comparison(t, k, train_models, val_configs):
        # Find Baseline Model for fold t
        base_model = next(m for m in train_models if m["t"] == t and m["k"] == -1)
        # Find Aug Model for fold t, source k
        aug_model = next(m for m in train_models if m["t"] == t and m["k"] == k)
        val_conf = val_configs[t]
        
        # Get u, v for Baseline
        u_base, v_base = get_uv_only(base_model, alpha_val, z_obs)
        # Get u, v for Augmented
        u_aug, v_aug = get_uv_only(aug_model, alpha_val, z_obs)
        
        # Compute Loss Coeffs
        C2_b, C1_b, C0_b = get_loss_coefs(val_conf["a_val"], val_conf["b_val"], u_base, v_base, val_conf["X_val"])
        C2_a, C1_a, C0_a = get_loss_coefs(val_conf["a_val"], val_conf["b_val"], u_aug, v_aug, val_conf["X_val"])
        
        return solve_quadratic_interval(C2_a - C2_b, C1_a - C1_b, C0_a - C0_b, z_obs)

    # Parallelize the K*T comparisons
    tasks = []
    for t in range(T):
        for k in range(K):
            tasks.append((t, k))
            
    results = Parallel(n_jobs=-1, prefer="threads")(
        delayed(solve_single_val_comparison)(t, k, train_models, val_configs) for t, k in tasks
    )
    
    L_final = max([res[0] for res in results])
    R_final = min([res[1] for res in results])
    return L_final, R_final
    

def get_uv_only(model_dict, alpha_val, z_obs):
    X, a, b = model_dict["X"], model_dict["a"], model_dict["b"]
    n, p = X.shape
    y = a + b * z_obs
    
    # Use warm start or low iter? No, needs precision.
    clf = Lasso(alpha=alpha_val, fit_intercept=False, tol=1e-8, max_iter=100000)
    clf.fit(X, y.flatten())
    
    active = np.flatnonzero(clf.coef_)
    u_full = np.zeros((p, 1))
    v_full = np.zeros((p, 1))
    
    if len(active) > 0:
        X_M = X[:, active]
        s_M = np.sign(clf.coef_[active]).reshape(-1, 1)
        lambda_val = n * alpha_val
        Gram_inv = np.linalg.pinv(X_M.T @ X_M)
        u_active = Gram_inv @ (X_M.T @ a - lambda_val * s_M)
        v_active = Gram_inv @ (X_M.T @ b)
        u_full[active] = u_active
        v_full[active] = v_active
        
    return u_full, v_full

import numpy as np
from sklearn.linear_model import Lasso

class WarmStartManager:
    """
    Holds persistent Lasso objects to enable Warm Starting.
    """
    def __init__(self, train_configs, alpha_val):
        self.models = {}
        self.configs = train_configs
        self.alpha = alpha_val
        
        # Initialize one Lasso object per configuration
        for conf in train_configs:
            key = (conf["t"], conf["k"]) # Unique ID for (Fold, Source)
            # warm_start=True is the Magic Switch
            self.models[key] = Lasso(
                alpha=alpha_val, 
                fit_intercept=False, 
                warm_start=True,  # <--- CRITICAL
                tol=1e-8, 
                max_iter=100000,
                selection='cyclic' # Cyclic is often faster for fine-tuning warm starts
            )

    def update_and_solve(self, zk):
        """
        Updates y(z) for all models and refits them instantly.
        Returns the computed intervals.
        """
        L_final, R_final = -np.inf, np.inf
        
        # We need to store u, v for the validation step
        model_solutions = {} 

        for conf in self.configs:
            key = (conf["t"], conf["k"])
            clf = self.models[key]
            
            # 1. Update Target y based on new z
            # y(z) = a + b * z
            y_new = conf["a"] + conf["b"] * zk
            
            # 2. Warm Start Fit (Extremely Fast)
            clf.fit(conf["X"], y_new.flatten())
            
            # 3. Compute Stability Interval (Algebra)
            L_model, R_model = self._compute_interval_algebra(clf, conf, self.alpha)
            
            # Aggregate Train Interval
            L_final = max(L_final, L_model)
            R_final = min(R_final, R_model)
            
            # Store solution for Validation step
            model_solutions[key] = self._get_uv(clf, conf)

        return L_final, R_final, model_solutions

    def _get_uv(self, clf, conf):
        """Helper to get u, v vectors from a fitted model"""
        active = np.flatnonzero(clf.coef_)
        n, p = conf["X"].shape
        u = np.zeros((p, 1))
        v = np.zeros((p, 1))
        
        if len(active) > 0:
            X = conf["X"]
            X_M = X[:, active]
            s_M = np.sign(clf.coef_[active]).reshape(-1, 1)
            
            lambda_val = n * self.alpha
            # Note: For speed, you could cache (X_M.T @ X_M)^-1 if active set doesn't change
            # But pinv is fast enough for p=100
            Gram_inv = np.linalg.pinv(X_M.T @ X_M)
            
            u[active] = Gram_inv @ (X_M.T @ conf["a"] - lambda_val * s_M)
            v[active] = Gram_inv @ (X_M.T @ conf["b"])
            
        return u, v

    def _compute_interval_algebra(self, clf, conf, alpha_val):
        # ... (Same algebra logic as compute_lasso_interval) ...
        # Copy the algebraic part of compute_lasso_interval here
        # Use clf.coef_ directly. Do NOT re-fit.
        X, a, b = conf["X"], conf["a"], conf["b"]
        n = X.shape[0]
        
        active_indices = np.flatnonzero(clf.coef_)
        inactive_indices = np.flatnonzero(clf.coef_ == 0)
        m = len(active_indices)
        lambda_val = n * alpha_val
        
        L, R = -np.inf, np.inf

        if m == 0:
            p = (1/lambda_val) * X.T @ a
            q = (1/lambda_val) * X.T @ b
            A = np.concatenate([q.flatten(), -q.flatten()])
            B = np.concatenate([(1 - p).flatten(), (1 + p).flatten()])
        else:
            X_M = X[:, active_indices]
            X_Mc = X[:, inactive_indices]
            s_M = np.sign(clf.coef_[active_indices]).reshape(-1, 1)
            
            Gram_inv = np.linalg.pinv(X_M.T @ X_M)
            u = Gram_inv @ (X_M.T @ a - lambda_val * s_M)
            v = Gram_inv @ (X_M.T @ b)
            
            R_a = a - X_M @ u
            R_b = b - X_M @ v
            p = (1/lambda_val) * X_Mc.T @ R_a
            q = (1/lambda_val) * X_Mc.T @ R_b
            
            A1 = -s_M * v
            B1 = s_M * u
            A2 = q
            B2 = np.ones((len(inactive_indices), 1)) - p
            A3 = -q
            B3 = np.ones((len(inactive_indices), 1)) + p
            
            A = np.concatenate([A1.flatten(), A2.flatten(), A3.flatten()])
            B = np.concatenate([B1.flatten(), B2.flatten(), B3.flatten()])

        pos_idx = A > 1e-9
        if np.any(pos_idx): R = np.min(B[pos_idx] / A[pos_idx])
        neg_idx = A < -1e-9
        if np.any(neg_idx): L = np.max(B[neg_idx] / A[neg_idx])
            
        return L, R

# --- FAST VALIDATION ---
def get_Z_val_warm(model_solutions, val_configs, z_obs, K, T, get_loss_coefs_func, solve_quad_func):
    """
    Computes Validation Stability using the u,v solutions from the WarmStartManager.
    No fitting required here! Pure algebra.
    """
    L_final, R_final = -np.inf, np.inf
    
    for t in range(T):
        # Baseline u, v
        u_base, v_base = model_solutions[(t, -1)]
        val_conf = val_configs[t]
        
        # Baseline Loss Coeffs
        C2_b, C1_b, C0_b = get_loss_coefs_func(val_conf["a_val"], val_conf["b_val"], u_base, v_base, val_conf["X_val"])
        
        for k in range(K):
            # Augmented u, v
            u_aug, v_aug = model_solutions[(t, k)]
            
            # Aug Loss Coeffs
            C2_a, C1_a, C0_a = get_loss_coefs_func(val_conf["a_val"], val_conf["b_val"], u_aug, v_aug, val_conf["X_val"])
            
            # Solve Quadratic
            L_vote, R_vote = solve_quad_func(C2_a - C2_b, C1_a - C1_b, C0_a - C0_b, z_obs)
            
            L_final = max(L_final, L_vote)
            R_final = min(R_final, R_vote)
            
    return L_final, R_final

#---#
import numpy as np
from sklearn.linear_model import Lasso

class WarmStartManager:
    """
    Holds persistent Lasso objects to enable Warm Starting.
    """
    def __init__(self, train_configs, alpha_val):
        self.models = {}
        self.configs = train_configs
        self.alpha = alpha_val
        
        # Initialize one Lasso object per configuration
        for conf in train_configs:
            key = (conf["t"], conf["k"]) # Unique ID for (Fold, Source)
            # warm_start=True is the Magic Switch
            self.models[key] = Lasso(
                alpha=alpha_val, 
                fit_intercept=False, 
                warm_start=True,  # <--- CRITICAL
                tol=1e-8, 
                max_iter=100000,
                selection='cyclic' # Cyclic is often faster for fine-tuning warm starts
            )

    def update_and_solve(self, zk):
        """
        Updates y(z) for all models and refits them instantly.
        Returns the computed intervals.
        """
        L_final, R_final = -np.inf, np.inf
        
        # We need to store u, v for the validation step
        model_solutions = {} 

        for conf in self.configs:
            key = (conf["t"], conf["k"])
            clf = self.models[key]
            
            # 1. Update Target y based on new z
            # y(z) = a + b * z
            y_new = conf["a"] + conf["b"] * zk
            
            # 2. Warm Start Fit (Extremely Fast)
            clf.fit(conf["X"], y_new.flatten())
            
            # 3. Compute Stability Interval (Algebra)
            L_model, R_model = self._compute_interval_algebra(clf, conf, self.alpha)
            
            # Aggregate Train Interval
            L_final = max(L_final, L_model)
            R_final = min(R_final, R_model)
            
            # Store solution for Validation step
            model_solutions[key] = self._get_uv(clf, conf)

        return L_final, R_final, model_solutions

    def _get_uv(self, clf, conf):
        """Helper to get u, v vectors from a fitted model"""
        active = np.flatnonzero(clf.coef_)
        n, p = conf["X"].shape
        u = np.zeros((p, 1))
        v = np.zeros((p, 1))
        
        if len(active) > 0:
            X = conf["X"]
            X_M = X[:, active]
            s_M = np.sign(clf.coef_[active]).reshape(-1, 1)
            
            lambda_val = n * self.alpha
            # Note: For speed, you could cache (X_M.T @ X_M)^-1 if active set doesn't change
            # But pinv is fast enough for p=100
            Gram_inv = np.linalg.pinv(X_M.T @ X_M)
            
            u[active] = Gram_inv @ (X_M.T @ conf["a"] - lambda_val * s_M)
            v[active] = Gram_inv @ (X_M.T @ conf["b"])
            
        return u, v

    def _compute_interval_algebra(self, clf, conf, alpha_val):
        # ... (Same algebra logic as compute_lasso_interval) ...
        # Copy the algebraic part of compute_lasso_interval here
        # Use clf.coef_ directly. Do NOT re-fit.
        X, a, b = conf["X"], conf["a"], conf["b"]
        n = X.shape[0]
        
        active_indices = np.flatnonzero(clf.coef_)
        inactive_indices = np.flatnonzero(clf.coef_ == 0)
        m = len(active_indices)
        lambda_val = n * alpha_val
        
        L, R = -np.inf, np.inf

        if m == 0:
            p = (1/lambda_val) * X.T @ a
            q = (1/lambda_val) * X.T @ b
            A = np.concatenate([q.flatten(), -q.flatten()])
            B = np.concatenate([(1 - p).flatten(), (1 + p).flatten()])
        else:
            X_M = X[:, active_indices]
            X_Mc = X[:, inactive_indices]
            s_M = np.sign(clf.coef_[active_indices]).reshape(-1, 1)
            
            Gram_inv = np.linalg.pinv(X_M.T @ X_M)
            u = Gram_inv @ (X_M.T @ a - lambda_val * s_M)
            v = Gram_inv @ (X_M.T @ b)
            
            R_a = a - X_M @ u
            R_b = b - X_M @ v
            p = (1/lambda_val) * X_Mc.T @ R_a
            q = (1/lambda_val) * X_Mc.T @ R_b
            
            A1 = -s_M * v
            B1 = s_M * u
            A2 = q
            B2 = np.ones((len(inactive_indices), 1)) - p
            A3 = -q
            B3 = np.ones((len(inactive_indices), 1)) + p
            
            A = np.concatenate([A1.flatten(), A2.flatten(), A3.flatten()])
            B = np.concatenate([B1.flatten(), B2.flatten(), B3.flatten()])

        pos_idx = A > 1e-9
        if np.any(pos_idx): R = np.min(B[pos_idx] / A[pos_idx])
        neg_idx = A < -1e-9
        if np.any(neg_idx): L = np.max(B[neg_idx] / A[neg_idx])
            
        return L, R

# --- FAST VALIDATION ---
def get_Z_val_warm(model_solutions, val_configs, z_obs, K, T, get_loss_coefs_func, solve_quad_func):
    """
    Computes Validation Stability using the u,v solutions from the WarmStartManager.
    No fitting required here! Pure algebra.
    """
    L_final, R_final = -np.inf, np.inf
    
    for t in range(T):
        # Baseline u, v
        u_base, v_base = model_solutions[(t, -1)]
        val_conf = val_configs[t]
        
        # Baseline Loss Coeffs
        C2_b, C1_b, C0_b = get_loss_coefs_func(val_conf["a_val"], val_conf["b_val"], u_base, v_base, val_conf["X_val"])
        
        for k in range(K):
            # Augmented u, v
            u_aug, v_aug = model_solutions[(t, k)]
            
            # Aug Loss Coeffs
            C2_a, C1_a, C0_a = get_loss_coefs_func(val_conf["a_val"], val_conf["b_val"], u_aug, v_aug, val_conf["X_val"])
            
            # Solve Quadratic
            L_vote, R_vote = solve_quad_func(C2_a - C2_b, C1_a - C1_b, C0_a - C0_b, z_obs)
            
            L_final = max(L_final, L_vote)
            R_final = min(R_final, R_vote)
            
    return L_final, R_final