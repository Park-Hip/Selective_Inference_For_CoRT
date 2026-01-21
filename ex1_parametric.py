import numpy as np
import SI_CoRT
import CoRT_builder

import warnings
warnings.filterwarnings("ignore")

def parametric(iteration, n_target, n_source, p, K, Ka, h, alpha, T, s_len, s_vector):
    para_results_storage = []
    CoRT_model = CoRT_builder.CoRT()

    cnt1 = 0
    cnt2 = 0
    cnt3 = 0
    cnt4 = 0

    for i in range(iteration):
        target_data, source_data = CoRT_model.gen_data(n_target, n_source, p, K, Ka, h, s_vector, s_len, "AR")
        result_dict = SI_CoRT.SI_parametric(n_target, p, K, target_data, source_data, T, s_len)
        if i % 50 == 0:
            print(f"Iteration {i}")
        if result_dict != None:
            # cnt1 += (result_dict['is_signal'] == True)
            # cnt2 += (result_dict['is_signal'] == False)
            # cnt3 += (result_dict['is_signal'] == True and result_dict['p_value'] <= alpha)
            # cnt4 += (result_dict['is_signal'] == False and result_dict['p_value'] <= alpha)
            # if i % 1 == 0:
            #     print(f"is_signal : {result_dict['is_signal']}, p_values[{i}]: {result_dict['p_value']}")
            #     print(f"FPR: {cnt4 / cnt2}, TPR: {cnt3 / cnt1}")
            #     print(f"is_not_signal: {int(cnt2), int(cnt4)}")
            #     print(f"is_signal: {int(cnt1), int(cnt3)}")
            #     # print("\n")
            #     print("===========================================================================")
                
            para_results_storage.append(result_dict)
    is_signal_cases = [r for r in para_results_storage if r['is_signal']]
    not_signal_cases = [r for r in para_results_storage if not r['is_signal']]

    false_positives = sum(1 for c in not_signal_cases if c['p_value'] <= alpha)
    fpr = false_positives / len(not_signal_cases)
    true_positives = sum(1 for r in is_signal_cases if r['p_value'] <= alpha)
    tpr = true_positives / len(is_signal_cases)

    return fpr, tpr