import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

import utils
import parametric
import over_conditioning
from helper import get_target_data, get_source_data, estimate_Sigma
import CoRT_builder
from sklearn.linear_model import Lasso
import numpy as np
from scipy.stats import norm

import warnings
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)

def SI_parametric_empirical(p, K, T, CONST_C, beta_index_list, target_data, source_data, n_target = 50, n_source = 100):
    Sigma = estimate_Sigma(target_data["X"], target_data["y"], n_target)

    CoRT_model = CoRT_builder.CoRT()
    similar_source_index = CoRT_model.find_similar_source(p, n_target, K, target_data, source_data, T=T, verbose=False)
    X_combined, y_combined = CoRT_model.prepare_CoRT_data(similar_source_index, source_data, target_data)

    N = X_combined.shape[0]
    lamda = CONST_C 

    model = Lasso(alpha=lamda / N, fit_intercept=False, tol=1e-14, max_iter = 1000000)
    model.fit(X_combined, y_combined.ravel())
    beta_hat_target = model.coef_[-p:]

    threshold = 1e-9
    M_obs = np.array([i for i, b in enumerate(beta_hat_target) if np.abs(b) > threshold])

    if len(M_obs) == 0:
        print(f"Lasso selected no features. Skipping.")

    print(f"M_obs = {M_obs}")

    p_values = []
    for index in beta_index_list:
        if index not in M_obs:
            print(f"Selected feature {index} is not in M_obs. Skipping.")
            p_values.append(None)
            continue
        else:
            for i, f in enumerate(M_obs):
                if f == index:
                    j = i
                    break

            X_target = target_data["X"]
            y_target = target_data["y"]

            active_cols = [idx for idx, coef in enumerate(beta_hat_target) if abs(coef) > 0]
            inactive_cols = [idx for idx, coef in enumerate(beta_hat_target) if abs(coef) <= 0]

            X_active = X_target[:, active_cols]
            X_inactive = X_target[:, inactive_cols]

            etaj, etajTy = utils.construct_test_statistic(y_target, j, X_active)

            b_global = Sigma @ etaj @ np.linalg.pinv(etaj.T @ Sigma @ etaj)
            a_global = (Sigma - b_global @ etaj.T) @ y_target

            folds = utils.split_target(T, X_target, y_target, n_target)

            tn_sigma = (np.sqrt(etaj.T @ Sigma @ etaj)).item()
            z_min = -20  * tn_sigma
            z_max = 20 * tn_sigma
            z_interval = parametric.solve_truncation_CoRT(z_min, z_max, X_target, folds, source_data, a_global, b_global, p, K, T, M_obs)
            p_value = utils.pivot(z_interval, etaj, etajTy, 0, Sigma)
            p_values.append(p_value)
    
    return p_values

def SI_over_conditioning_empirical(p, K, T, CONST_C, beta_index_list, target_data, source_data, n_target = 50, n_source = 100):
    CoRT_model = CoRT_builder.CoRT()

    Sigma = estimate_Sigma(target_data["X"], target_data["y"], n_target)

    similar_source_index = CoRT_model.find_similar_source(p, n_target, K, target_data, source_data, T=T, verbose=False)
    X_combined, y_combined = CoRT_model.prepare_CoRT_data(similar_source_index, source_data, target_data)

    N = X_combined.shape[0]
    lamda = CONST_C * np.sqrt(np.log(p)/ N)

    model = Lasso(alpha=lamda, fit_intercept=False, tol=1e-14, max_iter=1000000)
    model.fit(X_combined, y_combined.ravel())
    beta_hat_target = model.coef_[-p:]

    M_obs = np.array([i for i, b in enumerate(beta_hat_target) if b != 0])
    
    if len(M_obs) == 0:
        print(f"Lasso selected no features. Skipping.")
        return None

    print(f"M_obs = {M_obs}")

    p_values = []
    for index in beta_index_list:
        if index not in M_obs:
            print(f"Selected feature {index} is not in M_obs. Skipping.")
            p_values.append(None)
            continue
        else:
            for i, f in enumerate(M_obs):
                if f == index:
                    j = i
                    break

            X_target = target_data["X"]
            y_target = target_data["y"]
            X_active, X_inactive = utils.get_active_X(beta_hat_target, X_target)
            etaj, etajTy = utils.construct_test_statistic(y_target, j, X_active)

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
            p_values.append(p_value)

    return p_values


def data_splitting_empirical(p, K, T, CONST_C, beta_index_list, target_data, source_data, n_target = 50, n_source = 100):
    CoRT_model = CoRT_builder.CoRT()

    X_target = target_data["X"]
    y_target = target_data["y"]
    folds = utils.split_target(2, X_target, y_target, n_target)
    target_data_train = folds[0]
    target_data_test = folds[1]
    n = int(n_target / 2)

    similar_source_index = CoRT_model.find_similar_source(p, n, K, target_data_train, source_data, T, verbose=False)
    X_combined, y_combined = CoRT_model.prepare_CoRT_data(similar_source_index, source_data, target_data_train)
    N = X_combined.shape[0]
    lamda = CONST_C * np.sqrt(np.log(p) / N)
    model = Lasso(alpha=lamda, fit_intercept=False, tol=1e-12, max_iter=100000)
    model.fit(X_combined, y_combined.ravel())
    beta_hat_target = model.coef_[-p:]

    M_obs = np.array([i for i, b in enumerate(beta_hat_target) if b != 0])
    if len(M_obs) == 0:
        print(f"Lasso selected no features. Skipping.")

    print(f"M_obs = {M_obs}")

    p_values = []
    for index in beta_index_list:
        if index not in M_obs:
            print(f"Selected feature {index} is not in M_obs. Skipping.")
            p_values.append(None)
            continue
        else:
            for i, f in enumerate(M_obs):
                if f == index:
                    j = i
                    break 

            X_target = target_data_test["X"]
            y_target = target_data_test["y"]
            n_target_test = X_target.shape[0]
            X_active, X_inactive = utils.get_active_X(beta_hat_target, X_target)
            etaj, etajTy = utils.construct_test_statistic(y_target, j, X_active)
            Sigma = estimate_Sigma(X_target, y_target, n_target_test)
            tn_sigma = (np.sqrt(etaj.T @ Sigma @ etaj)).item()

            p_value = 2 * (1 - norm.cdf(abs(etajTy), loc=0, scale=tn_sigma))
            p_values.append(p_value)

    return p_values

def bonferroni_empirical(p, K, T, CONST_C, beta_index_list, target_data, source_data, n_target = 50, n_source = 100):
    CoRT_model = CoRT_builder.CoRT()

    X_target = target_data["X"]
    y_target = target_data["y"]
    similar_source_index = CoRT_model.find_similar_source(p, n_target, K, target_data, source_data, T, verbose=False)
    X_combined, y_combined = CoRT_model.prepare_CoRT_data(similar_source_index, source_data, target_data)

    n = X_combined.shape[0]
    lamda = CONST_C * np.sqrt(np.log(p) / n)
    model = Lasso(alpha=lamda, fit_intercept=False, tol=1e-12, max_iter=100000)
    model.fit(X_combined, y_combined.ravel())
    beta_hat_target = model.coef_[-p:]

    M_obs = np.array([i for i, b in enumerate(beta_hat_target) if b != 0])

    if len(M_obs) == 0:
        print("Lasso selected no features. Skipping.")
        return None
    
    print(f"M_obs = {M_obs}")

    p_values = []
    for index in beta_index_list:
        if index not in M_obs:
            print(f"Selected feature {index} is not in M_obs. Skipping.")
            p_values.append(None)
            continue
        else:
            for i, f in enumerate(M_obs):
                if f == index:
                    j = i
                    break

            X_active, X_inactive = utils.get_active_X(beta_hat_target, X_target)

            etaj, etajTy = utils.construct_test_statistic(y_target, j, X_active)
            Sigma = estimate_Sigma(target_data["X"], target_data["y"], n_target)
            sigma_squared = (etaj.T @ Sigma @ etaj).item()
            sigma_z = np.sqrt(sigma_squared)

            naive_p_value = 2 * (1 - norm.cdf(abs(etajTy), loc = 0, scale = sigma_z))

            bonferroni_p_value = min(1.0, naive_p_value * p)
            p_values.append(bonferroni_p_value)

    return p_values