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












