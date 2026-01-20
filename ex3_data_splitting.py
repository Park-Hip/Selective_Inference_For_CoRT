import numpy as np
import CoRT_builder
import utils
from sklearn.linear_model import Lasso
from scipy.stats import norm

def data_splitting(iteration, n_target, n_source, p, K, Ka, h, alpha, T, s_len, s_vector):
    N = n_target / 2 + Ka * n_source
    NI = n_target / 2 + n_source
    lamda_k_source = 1.2 * np.sqrt(np.log(p)/ N)
    lamda_1_source = 1.2 * np.sqrt(np.log(p)/ NI) 
    lamda_not_source = 1.2 * np.sqrt(np.log(p) / (n_target / 2)) 

    CoRT_model = CoRT_builder.CoRT(lamda_not_source)
    para_results_storage = []

    for i in range(0, iteration):
        target_data, source_data = CoRT_model.gen_data(n_target, n_source, p, K, Ka, h, s_vector, s_len, "AR")
        # splitting data
        if i % 50 == 0:
            print(f"Iteration {i}")
        X_target = target_data["X"]
        y_target = target_data["y"]
        folds = utils.split_target(2, X_target, y_target, n_target)
        target_data_train = folds[0]
        target_data_test = folds[1]
        n_target_new = int(n_target / 2)

        similar_source_index = CoRT_model.find_similar_source(n_target_new, K, target_data_train, source_data, lamda_not_source, lamda_1_source, T=T, verbose=False)
        X_combined, y_combined = CoRT_model.prepare_CoRT_data(similar_source_index, source_data, target_data_train)
        model = Lasso(alpha=lamda_k_source, fit_intercept=False, tol=1e-10, max_iter=100000)
        model.fit(X_combined, y_combined.ravel())
        beta_hat_target = model.coef_[-p:]

        M_obs = np.array([i for i, b in enumerate(beta_hat_target) if b != 0])
        if len(M_obs) == 0:
            print(f"Iteration {iter}: Lasso selected no features. Skipping.")
            continue

        j = np.random.choice(len(M_obs))
        selected_feature_index = M_obs[j]

        X_target = target_data_test["X"]
        y_target = target_data_test["y"]
        X_active, X_inactive = utils.get_active_X(beta_hat_target, X_target)
        etaj, etajTy = utils.construct_test_statistic(y_target, j, X_active)
        Sigma = np.eye(n_target_new)
        tn_sigma = (np.sqrt(etaj.T @ Sigma @ etaj)).item()

        p_value = 2 * (1 - norm.cdf(abs(etajTy), loc = 0, scale = tn_sigma))

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