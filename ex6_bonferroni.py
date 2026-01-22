import numpy as np
import CoRT_builder
import utils
from sklearn.linear_model import Lasso
from scipy.stats import norm

CONST_C = 1.1

def bonferroni(iteration, n_target, n_source, p, K, Ka, h, alpha, T, s_len, s_vector):
    CoRT_model = CoRT_builder.CoRT()
    para_results_storage = []

    for i in range(iteration):
        if i % 50 == 0:
            print(f"Iteration {i}")
        target_data, source_data = CoRT_model.gen_data(n_target, n_source, p, K, Ka, h, s_vector, s_len, "AR")
        CoRT_model = CoRT_builder.CoRT()
        similar_source_index = CoRT_model.find_similar_source(p, n_target, K, target_data, source_data, T=T, verbose=False)
        X_combined, y_combined = CoRT_model.prepare_CoRT_data(similar_source_index, source_data, target_data)

        n = X_combined.shape[0]
        lamda = CONST_C * np.sqrt(np.log(p) / n)
        model = Lasso(alpha=lamda, fit_intercept=False, tol=1e-12, max_iter=1000000)
        model.fit(X_combined, y_combined.ravel())
        beta_hat_target = model.coef_[-p:]

        M_obs = np.array([i for i, b in enumerate(beta_hat_target) if b != 0])

        if len(M_obs) == 0:
            print("Lasso selected no features. Skipping.")
            continue
        j = np.random.choice(len(M_obs))
        selected_feature_index = M_obs[j]

        X_target = X_combined
        y_target = y_combined

        X_active, X_inactive = utils.get_active_X(beta_hat_target, X_target)

        etaj, etajTy = utils.construct_test_statistic(y_target, j, X_active)
        Sigma = np.eye(n)
        sigma_squared = (etaj.T @ Sigma @ etaj).item()
        sigma_z = np.sqrt(sigma_squared)
        
        naive_p_value = 2 * (1 - norm.cdf(abs(etajTy), loc = 0, scale = sigma_z))
        
        bonferroni_p_value = min(1.0, naive_p_value * p)
        
        is_signal = (selected_feature_index < s_len) 
        
        result_dict = {
            "p_value": bonferroni_p_value, # Trả về p-value ĐÃ hiệu chỉnh
            "is_signal": is_signal,
            "feature_idx": selected_feature_index
        }
        para_results_storage.append(result_dict)

    is_signal_cases = [r for r in para_results_storage if r['is_signal']]
    not_signal_cases = [r for r in para_results_storage if not r['is_signal']]

    false_positives = sum(1 for c in not_signal_cases if c['p_value'] <= alpha)
    fpr = false_positives / len(not_signal_cases)
    true_positives = sum(1 for r in is_signal_cases if r['p_value'] <= alpha)
    tpr = true_positives / len(is_signal_cases)

    return fpr, tpr

