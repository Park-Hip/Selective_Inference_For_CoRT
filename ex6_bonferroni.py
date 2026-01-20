import numpy as np
import CoRT_builder
import utils
from sklearn.linear_model import Lasso
from scipy.stats import norm

def bonferroni(iteration, n_target, n_source, p, K, Ka, h, alpha, T, s_len, s_vector):
    N = n_target + Ka * n_source
    NI = n_target + n_source
    lamda_k_source = 1.2 * np.sqrt(np.log(p)/ N)
    lamda_1_source = 1.2 * np.sqrt(np.log(p)/ NI) 
    lamda_not_source = 1.2 * np.sqrt(np.log(p) / n_target) 

    CoRT_model = CoRT_builder.CoRT(lamda_not_source)
    para_results_storage = []

    for i in range(iteration):
        target_data, source_data = CoRT_model.gen_data(n_target, n_source, p, K, Ka, h, s_vector, s_len, "AR")
        CoRT_model = CoRT_builder.CoRT(alpha=lamda_not_source)
        similar_source_index = CoRT_model.find_similar_source(n_target, K, target_data, source_data, lamda_not_source, lamda_1_source, T=T, verbose=False)
        X_combined, y_combined = CoRT_model.prepare_CoRT_data(similar_source_index, source_data, target_data)

        model = Lasso(alpha=lamda_k_source, fit_intercept=False, tol=1e-10, max_iter=100000)
        model.fit(X_combined, y_combined.ravel())
        beta_hat_target = model.coef_[-p:]

        # Tập biến được chọn
        M_obs = np.array([i for i, b in enumerate(beta_hat_target) if b != 0])

        if len(M_obs) == 0:
            print("Lasso selected no features. Skipping.")
            continue

        # 2. Chọn ngẫu nhiên 1 biến để kiểm định (theo cách code của bạn)
        j = np.random.choice(len(M_obs))
        selected_feature_index = M_obs[j]

        X_target = X_combined
        y_target = y_combined
        
        # Lấy ma trận X active cho thống kê kiểm định
        X_active, X_inactive = utils.get_active_X(beta_hat_target, X_target)

        # 3. Tính toán thống kê Naive (Không có Selective Inference)
        # etajTy ~ N(mu, sigma^2)
        etaj, etajTy = utils.construct_test_statistic(y_target, j, X_active)
        n_combined = X_target.shape[0]
        Sigma = np.eye(n_combined)
        sigma_squared = (etaj.T @ Sigma @ etaj).item()
        sigma_z = np.sqrt(sigma_squared)
        
        # Tính Naive P-value (2-sided Z-test)
        # P(|Z| > |z_obs|) = 2 * (1 - CDF(|z_obs|))
        naive_p_value = 2 * (1 - norm.cdf(abs(etajTy), loc = 0, scale = sigma_z))
        
        # 4. Hiệu chỉnh Bonferroni
        # Nhân p-value với tổng số chiều p (để kiểm soát FWER trên toàn bộ không gian tìm kiếm)
        # Đây là cách tiếp cận bảo thủ nhất.
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

