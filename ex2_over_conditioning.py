import numpy as np
import SI_CoRT
import CoRT_builder

def over_conditioning(iteration, n_target, n_source, p, K, Ka, h, alpha, T, s_len, s_vector): 
    CoRT_model = CoRT_builder.CoRT()
    para_results_storage = []

    for i in range(iteration):
        target_data, source_data = CoRT_model.gen_data(n_target, n_source, p, K, Ka, h, s_vector, s_len, "AR")
        result_dict = SI_CoRT.SI_over_conditioning(n_target, p, K, target_data, source_data, T, s_len)
        # print(i, result_dict['p_value'])
        if result_dict != None:
            # if i % 50 == 0:
            #     print(f"is_signal : {result_dict['is_signal']}, p_value[{i}] = {result_dict['p_value']}")
            #     print("==========================================================================================")
            para_results_storage.append(result_dict)
        if i % 50 == 0:
            print(f"Iteration {i}")

    is_signal_cases = [r for r in para_results_storage if r['is_signal']]
    not_signal_cases = [r for r in para_results_storage if not r['is_signal']]

    false_positives = sum(1 for c in not_signal_cases if c['p_value'] <= alpha)
    fpr = false_positives / len(not_signal_cases)
    true_positives = sum(1 for r in is_signal_cases if r['p_value'] <= alpha)
    tpr = true_positives / len(is_signal_cases)
    return fpr, tpr