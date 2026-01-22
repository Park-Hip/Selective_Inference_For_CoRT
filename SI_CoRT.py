import numpy as np
from sklearn.linear_model import Lasso

import utils
import parametric
import over_conditioning
import CoRT_builder 

CONST_C = 1.5

def SI_parametric(n_target, p, K, target_data, source_data, T, s_len):
    CoRT_model = CoRT_builder.CoRT()
    similar_source_index = CoRT_model.find_similar_source(p, n_target, K, target_data, source_data, T=T, verbose=False)
    X_combined, y_combined = CoRT_model.prepare_CoRT_data(similar_source_index, source_data, target_data)

    N = X_combined.shape[0]
    lamda = CONST_C * np.sqrt(np.log(p)/ N)

    model = Lasso(alpha=lamda, fit_intercept=False, tol=1e-12, max_iter = 1000000)
    model.fit(X_combined, y_combined.ravel())
    beta_hat_target = model.coef_[-p:]

    M_obs = np.array([i for i, b in enumerate(beta_hat_target) if b != 0])

    if len(M_obs) == 0:
        print("Lasso selected no features. Skipping.")
        return None

    j = np.random.choice(len(M_obs))
    selected_feature_index = M_obs[j]

    X_target = target_data["X"]
    y_target = target_data["y"]
    X_active, X_inactive = utils.get_active_X(beta_hat_target, X_target)

    etaj, etajTy = utils.construct_test_statistic(y_target, j, X_active)

    Sigma = np.eye(n_target)
    b_global = Sigma @ etaj @ np.linalg.pinv(etaj.T @ Sigma @ etaj)
    a_global = (Sigma - b_global @ etaj.T) @ y_target

    folds = utils.split_target(T, X_target, y_target, n_target)
    
    tn_sigma = (np.sqrt(etaj.T @ Sigma @ etaj)).item()
    z_min = -20  * tn_sigma
    z_max = 20 * tn_sigma
    z_interval = parametric.solve_truncation_CoRT(z_min, z_max, X_target, folds, source_data, a_global, b_global, p, K, T, M_obs)
    p_value = utils.pivot(z_interval, etaj, etajTy, 0, Sigma)
    
    is_signal = (selected_feature_index < s_len) 
    result_dict = {
        "p_value": p_value,
        "is_signal": is_signal,
        "feature_idx": selected_feature_index
    }
    
    return result_dict

def SI_over_conditioning(n_target, p, K, target_data, source_data, T, s_len):
    CoRT_model = CoRT_builder.CoRT()
    similar_source_index = CoRT_model.find_similar_source(p, n_target, K, target_data, source_data, T=T, verbose=False)
    X_combined, y_combined = CoRT_model.prepare_CoRT_data(similar_source_index, source_data, target_data)

    N = X_combined.shape[0]
    lamda = CONST_C * np.sqrt(np.log(p)/ N)

    model = Lasso(alpha=lamda, fit_intercept=False, tol=1e-12, max_iter = 1000000)
    model.fit(X_combined, y_combined.ravel())
    beta_hat_target = model.coef_[-p:]

    M_obs = np.array([i for i, b in enumerate(beta_hat_target) if b != 0])

    if len(M_obs) == 0:
        print(f"Iteration {iter}: Lasso selected no features. Skipping.")
        return None

    j = np.random.choice(len(M_obs))
    selected_feature_index = M_obs[j]

    X_target = target_data["X"]
    y_target = target_data["y"]
    X_active, X_inactive = utils.get_active_X(beta_hat_target, X_target)
    etaj, etajTy = utils.construct_test_statistic(y_target, j, X_active)

    Sigma = np.eye(n_target)
    b_global = Sigma @ etaj @ np.linalg.pinv(etaj.T @ Sigma @ etaj)
    a_global = (np.eye(n_target) - b_global @ etaj.T) @ y_target

    folds = utils.split_target(T, X_target, y_target, n_target)

    L_base_agu, R_base_agu = over_conditioning.get_Z_base_aug(p, etajTy, folds, source_data, a_global, b_global, K, T)
    L_val, R_val = over_conditioning.get_Z_val(p, folds, T, K, a_global, b_global, etajTy, source_data)
    L_CoRT, R_CoRT, Az = over_conditioning.get_Z_CoRT(p, X_combined, similar_source_index, a_global, b_global, source_data, etajTy)

    L_final, R_final = utils.combine_Z(L_base_agu, R_base_agu, L_val, R_val, L_CoRT, R_CoRT)

    etaT_sigma_eta = (etaj.T @ Sigma @ etaj).item()
    sigma_z = np.sqrt(etaT_sigma_eta)
    truncated_cdf = utils.computed_truncated_cdf(L_final, R_final, etajTy, 0, sigma_z)
    p_value = 2 * min(truncated_cdf, 1 - truncated_cdf)
    is_signal = (selected_feature_index < s_len) 
    result_dict = {
        "p_value": p_value,
        "is_signal": is_signal,
        "feature_idx": selected_feature_index
    }
    return result_dict
    
