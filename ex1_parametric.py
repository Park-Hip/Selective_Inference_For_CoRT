import numpy as np
import SI_CoRT
import CoRT_builder

import warnings
warnings.filterwarnings("ignore")

def parametric(iteration, n_target, n_source, p, K, Ka, h, alpha, T, s_len, s_vector):
    N = n_target + Ka * n_source
    NI = n_target + n_source
    # lamda_k_source = 1.2 * np.sqrt(np.log(p)/ N)
    # lamda_1_source = 1.2 * np.sqrt(np.log(p)/ NI) 
    # lamda_not_source = 1.2 * np.sqrt(np.log(p) / n_target) 

    lamda_k_source = 2 * np.sqrt(np.log(p)/ N)
    lamda_1_source = 2 * np.sqrt(np.log(p)/ NI) 
    lamda_not_source = 2 * np.sqrt(np.log(p) / n_target) 

    para_results_storage = []
    CoRT_model = CoRT_builder.CoRT(alpha=lamda_not_source)

    cnt1 = 0
    cnt2 = 0
    cnt3 = 0
    cnt4 = 0

    for i in range(iteration):
        target_data, source_data = CoRT_model.gen_data(n_target, n_source, p, K, Ka, h, s_vector, s_len, "AR")
        result_dict = SI_CoRT.SI_parametric(n_target, p, K, target_data, source_data, lamda_not_source, lamda_1_source, lamda_k_source, T, s_len)
        if result_dict != None:
            para_results_storage.append(result_dict)
        if (i % 50) == 0:
            print(f"Iteration {i}")
    is_signal_cases = [r for r in para_results_storage if r['is_signal']]
    not_signal_cases = [r for r in para_results_storage if not r['is_signal']]

    false_positives = sum(1 for c in not_signal_cases if c['p_value'] <= alpha)
    fpr = false_positives / len(not_signal_cases)
    true_positives = sum(1 for r in is_signal_cases if r['p_value'] <= alpha)
    tpr = true_positives / len(is_signal_cases)

    return fpr, tpr