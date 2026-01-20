import numpy as np
import CoRT_builder
import utils
from sklearn.linear_model import Lasso
from scipy.stats import norm


def cort_without_inference(iteration, n_target, n_source, p, K, Ka, h, alpha, T, s_len, s_vector):
    N = n_target + Ka * n_source
    NI = n_target + n_source
    lamda_k_source = 1.2 * np.sqrt(np.log(p)/ N)
    lamda_1_source = 1.2 * np.sqrt(np.log(p)/ NI) 
    lamda_not_source = 1.2 * np.sqrt(np.log(p) / n_target) 

    CoRT_model = CoRT_builder.CoRT(lamda_not_source)
    para_results_storage = []
    
    for i in range(0, iteration):
        target_data, source_data = CoRT_model.gen_data(n_target, n_source, p, K, Ka, h, s_vector, s_len, "AR")
        if (i % 50) == 0:
            print(f"Iteration {i}")
        similar_source_index = CoRT_model.find_similar_source(n_target, K, target_data, source_data, lamda_not_source, lamda_1_source, T=T, verbose=False)
        X_combined, y_combined = CoRT_model.prepare_CoRT_data(similar_source_index, source_data, target_data)
        model = Lasso(alpha=lamda_k_source, fit_intercept=False, tol=1e-10, max_iter=100000)
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

    is_signal_cases = [r for r in para_results_storage if r['is_signal']]
    not_signal_cases = [r for r in para_results_storage if not r['is_signal']]

    false_positives = sum(1 for c in not_signal_cases if c['p_value'] <= alpha)
    fpr = false_positives / len(not_signal_cases)
    true_positives = sum(1 for r in is_signal_cases if r['p_value'] <= alpha)
    tpr = true_positives / len(is_signal_cases)
    return fpr, tpr
