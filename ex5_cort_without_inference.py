import numpy as np
import CoRT_builder
import utils
from sklearn.linear_model import Lasso
from scipy.stats import norm

CONST_C = 1.1

def cort_without_inference(iteration, n_target, n_source, p, K, Ka, h, alpha, T, s_len, s_vector):
    CoRT_model = CoRT_builder.CoRT(0)
    para_results_storage = []

    for i in range(0, iteration):
        target_data, source_data = CoRT_model.gen_data(n_target, n_source, p, K, Ka, h, s_vector, s_len, "AR")

        similar_source_index = CoRT_model.find_similar_source(p, n_target, K, target_data, source_data, T=T, verbose=False)
        X_combined, y_combined = CoRT_model.prepare_CoRT_data(similar_source_index, source_data, target_data)
        n = X_combined.shape[0]
        lamda = CONST_C * np.sqrt(np.log(p) / n)
        model = Lasso(alpha=lamda, fit_intercept=False, tol=1e-12, max_iter=1000000)
        model.fit(X_combined, y_combined.ravel())
        beta_hat_target = model.coef_[-p:]

        M_obs = np.array([i for i, b in enumerate(beta_hat_target) if b != 0])
        if len(M_obs) == 0:
            print(f"Iteration {iter}: Lasso selected no features. Skipping.")
            continue

        j = np.random.choice(len(M_obs))
        selected_feature_index = M_obs[j]

        p_value = 0.0

        is_signal = (selected_feature_index < s_len) 
        result_dict = {
            "p_value": p_value,
            "is_signal": is_signal,
            "feature_idx": selected_feature_index
        }
        # print(f"is_signal : {result_dict['is_signal']}, p_values[{i}]: {result_dict['p_value']}")
        para_results_storage.append(result_dict)
        if (i % 50 == 0):
            print(f"Iteration {i}")

    is_signal_cases = [r for r in para_results_storage if r['is_signal']]
    not_signal_cases = [r for r in para_results_storage if not r['is_signal']]

    false_positives = sum(1 for c in not_signal_cases if c['p_value'] <= alpha)
    fpr = false_positives / len(not_signal_cases)
    true_positives = sum(1 for r in is_signal_cases if r['p_value'] <= alpha)
    tpr = true_positives / len(is_signal_cases)
    return fpr, tpr
