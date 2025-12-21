import numpy as np
from sklearn.linear_model import Lasso
import matplotlib.pyplot as plt
import importlib
import warnings
from sklearn.exceptions import ConvergenceWarning
import copy

# Import your custom modules
from CoRT_builder import CoRT
import utils
import parametric_optim

# Reload modules to ensure latest changes are picked up
importlib.reload(utils)
importlib.reload(parametric_optim)

# --- LINE PROFILER SETUP ---
# This block allows the script to run as standard Python (without kernprof)
# preventing NameError on the @profile decorator.
try:
    profile
except NameError:
    profile = lambda x: x

@profile
def run_simulation():
    # --- CONFIGURATION ---
    n_target = 50
    n_source = 10
    p = 50
    K = 3
    Ka = 1
    h = 30
    lamda = 0.05
    s_vector = [0,0,0,0,0,0,0,0,0,0]
    T = 5
    s = len(s_vector)
    
    # --- REQUESTED CHANGE: ITERATION = 1 ---
    iteration = 1 

    CoRT_model = CoRT(alpha=lamda)
    p_values = []

    for step in range(iteration):
        print(f"--- Starting Iteration {step+1}/{iteration} ---")
        
        target_data, source_data = CoRT_model.gen_data(n_target, n_source, p, K, Ka, h, s_vector, s, "AR")
        similar_source_index = CoRT_model.find_similar_source(n_target, K, target_data, source_data, T=T, verbose=False)
        X_combined, y_combined = CoRT_model.prepare_CoRT_data(similar_source_index, source_data, target_data)

        model = Lasso(alpha=lamda, fit_intercept=False, tol=1e-10, max_iter=10000000)
        model.fit(X_combined, y_combined.ravel())
        beta_hat_target = model.coef_[-p:]

        active_indices = np.array([i for i, b in enumerate(beta_hat_target) if b != 0])

        if len(active_indices) == 0:
            print(f"Iteration {step}: Lasso selected no features. Skipping.")
            continue

        j = np.random.choice(len(active_indices))

        X_target = target_data["X"]
        y_target = target_data["y"]
        X_active, X_inactive = utils.get_active_X(beta_hat_target, X_target)

        etaj, etajTy = utils.construct_test_statistic(y_target, j, X_active)

        Sigma = np.eye(n_target)
        b_global = Sigma @ etaj @ np.linalg.pinv(etaj.T @ Sigma @ etaj)
        a_global = (Sigma - b_global @ etaj.T) @ y_target

        folds = utils.split_target(T, X_target, y_target, n_target)

        z_k = -20
        z_max = 20

        Z_train_list = parametric_optim.get_Z_train(z_k, folds, source_data, a_global, b_global, lamda, K, T)
        Z_val_list = parametric_optim.get_Z_val(z_k, folds, T, K, a_global, b_global, lamda, source_data)

        target_data_current = {"X": X_target, "y": a_global + z_k * b_global}
        similar_source_current = parametric_optim.find_similar_source(z_k, a_global, b_global, lamda,  n_target, K, target_data_current, source_data, T=T, verbose=False)
        X_combined_new, y_combined_new = CoRT_model.prepare_CoRT_data(similar_source_current, source_data, target_data_current)
        L_CoRT, R_CoRT, Az = parametric_optim.get_Z_CoRT(X_combined_new, similar_source_current, lamda, a_global, b_global, source_data, z_k)

        offset = p * len(similar_source_index)
        Az_target_only = np.array([idx - offset for idx in Az if idx >= offset])

        z_list = [z_k]
        Az_list = []

        print("="*100)
        print("Initialization")
        print(f"Initial similar source index: {similar_source_index}")
        print(f"z_obs: {etajTy:.5f}")
        print("="*100)

        step_count = 0
        stopper = "empty"

        while z_k < z_max:
            step_count += 1
            # print(f"zk at step {step_count}: {z_k:.5f}")
            
            current_num_sources = len(similar_source_current)
            offset = p * current_num_sources
            Az_target_current = np.array([idx - offset for idx in Az if idx >= offset])
            Az_list.append(Az_target_current)
            
            mn = z_max
            stopper = None

            for val in Z_train_list:
                if mn > val[4]:
                    mn = val[4]
                    stopper = f"TRAIN[type={val[0]}][t={val[1]}][k={val[2]}][L={val[3]:.5f}][R={val[4]:.5f}]"

            for val in Z_val_list:
                if mn > val[3]:
                    mn = val[3]
                    stopper = f"VAL[t={val[0]}][k={val[1]}][L={val[2]:.5f}][R={val[3]:.5f}]"

            if mn > R_CoRT:
                mn = R_CoRT
                stopper = f"CORT[{L_CoRT:.5f}, {R_CoRT:.5f}]"

            R_final = mn

            if R_final - z_k < -1e-9:
                # print("[WARNING] R_final is before zk")
                z_k += 0.001

            z_k = max(R_final, z_k) + 1e-5

            if (z_k >= z_max):
                z_list.append(z_max)
            else:
                z_list.append(z_k)

            update_train_needed = False
            update_val_needed = False
            update_cort_needed = False
            
            if stopper and "TRAIN" in stopper:
                update_train_needed = True
                update_val_needed = True   
                update_cort_needed = True

            elif stopper and "VAL" in stopper:
                update_val_needed = True
                update_cort_needed = True

            elif stopper and "CORT" in stopper:
                update_cort_needed = True

            if update_train_needed:
                for val in Z_train_list:
                    if val[4] <= z_k + 1e-9:
                        l, r = parametric_optim.update_Z_train(val, z_k, folds, source_data, a_global, b_global, lamda, K, T)
                        val[3] = l
                        val[4] = r

            if update_val_needed:
                for val in Z_val_list:
                    if val[3] <= z_k + 1e-9:
                        l, r = parametric_optim.update_Z_val(val, z_k, folds, T, K, a_global, b_global, lamda, source_data)
                        val[2] = l
                        val[3] = r

            if update_cort_needed:
                target_data_current = {"X": X_target, "y": a_global + z_k * b_global}
                similar_source_current = parametric_optim.find_similar_source(z_k, a_global, b_global, lamda, n_target, K, target_data_current, source_data, T=T, verbose=False)
                X_combined_new, y_combined_new = CoRT_model.prepare_CoRT_data(similar_source_current, source_data, target_data_current)
                L_CoRT, R_CoRT, Az = parametric_optim.get_Z_CoRT(X_combined_new, similar_source_current, lamda, a_global, b_global, source_data, z_k)

        # Post-loop interval processing
        z_interval = []
        for i in range(len(Az_list)):
            if np.array_equal(active_indices, Az_list[i]):
                    z_interval.append([z_list[i], z_list[i+1]]) 

        # Merge intervals
        new_z_interval = []
        for interval in z_interval:
            if len(new_z_interval) == 0:
                new_z_interval.append(interval)
            else:
                dif = abs(interval[0] - new_z_interval[-1][1])
                if dif < 0.001:
                    new_z_interval[-1][1] = interval[1]
                else:
                    new_z_interval.append(interval)
        z_interval = new_z_interval
        
        print(f"z_obs: {etajTy:.5f}")
        print(f"{len(z_interval)} intervals found: {z_interval}")

        p_value = parametric_optim.pivot(active_indices, Az_list, z_list, etaj, etajTy, 0, Sigma)
        p_values.append(p_value)
        print(f"Processing p-value: {p_value}")

    return p_values

if __name__ == "__main__":
    p_values = run_simulation() 