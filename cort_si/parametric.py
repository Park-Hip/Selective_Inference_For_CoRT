import numpy as np
import utils
import solve_interval
import CoRT_builder

def solve_truncation_interval(z_min, z_max, folds, source_data, a_global, b_global, lamda_not_source, lamda_1_source, K, T): 
    interval = {}
    for k in range(K):
        for t in range(T):
            fold_indices = []
            start = 0
            for f in folds:
                size = f["X"].shape[0]
                fold_indices.append(np.arange(start, start+size))
                start += size
            train_indices_list = []
            X_train_list = []
            
            for i in range(T):
                if i != t:
                    train_indices_list.append(fold_indices[i])
                    X_train_list.append(folds[i]["X"])

            train_indices = np.concatenate(train_indices_list)
            X_target_train = np.vstack(X_train_list)
            X_base, a_base, b_base = utils.get_affine_params(X_target_train, train_indices, a_global, b_global)

            z1 = z_min
            z1_max = z_max
            interval[(k, t)] = []
            while z1 < z1_max:
                L_base, R_base = solve_interval.compute_lasso_interval(X_base, a_base, b_base, lamda_not_source, z1)
                source_data_k = source_data[k]
                X_aug, a_aug, b_aug = utils.get_affine_params(X_target_train, train_indices, a_global, b_global, source_data_k)
                
                z2 = z1
                z2_max = min(R_base, z1_max)
                while z2 < z2_max:  
                    L_aug, R_aug = solve_interval.compute_lasso_interval(X_aug, a_aug, b_aug, lamda_1_source, z2)
                    z3 = z2
                    z3_max = min(R_aug, z2_max)
                    while z3 < z3_max:
                        u_base, v_base = utils.get_u_v(X_base, a_base, b_base, z3, lamda_not_source)
                        X_val = folds[t]["X"]
                        val_indices = fold_indices[t]
                        
                        _, a_base_val, b_base_val = utils.get_affine_params(X_val, val_indices, a_global, b_global)
                        C2_base, C1_base, C0_base = utils.get_loss_coefs(a_base_val, b_base_val, u_base, v_base, X_val)

                        u_aug, v_aug = utils.get_u_v(X_aug, a_aug, b_aug, z3, lamda_1_source)
                        C2_aug, C1_aug, C0_aug = utils.get_loss_coefs(a_base_val, b_base_val, u_aug, v_aug, X_val)

                        A_dif = C2_aug - C2_base
                        B_dif = C1_aug - C1_base
                        C_dif = C0_aug - C0_base

                        L_val, R_val = solve_interval.compute_quadratic_interval(A_dif, B_dif, C_dif, z3)
                        delta = A_dif * z3 * z3 + B_dif * z3 + C_dif
                        cnt = 0
                        if delta <= 0: 
                            cnt = 1
                        else:
                            cnt = 0
                        l = z3
                        r = min(R_val, z3_max)
                        interval[(k, t)].append([l, r, cnt])
                        z3 = max(z3, R_val) + 1e-5
                    z2 = max(R_aug, z2) + 1e-5
                z1 = max(z1, R_base) + 1e-5
    return interval
def solve_truncation_CoRT(z_min, z_max, X_target, folds, source_data, a_global, b_global, lamda_not_source, lamda_1_source, lamda_k_source, p, K, T, M_obs):
    interval = solve_truncation_interval(z_min, z_max, folds, source_data, a_global, b_global, lamda_not_source, lamda_1_source, K, T)
    CoRT_model = CoRT_builder.CoRT(alpha=lamda_k_source)
    z_k = z_min
    z_interval = []

    while z_k < z_max:
        L_final, R_final, similar_source_index = solve_interval.compute_similar_source(z_k, z_max, interval, K, T)
        target_data_current = {"X": X_target, "y": a_global + z_k * b_global}
        X_combined_new, y_combined_new = CoRT_model.prepare_CoRT_data(similar_source_index, source_data, target_data_current)
        L_CoRT, R_CoRT, Az = solve_interval.compute_lasso_interval(X_combined_new, a_global, b_global, lamda_k_source, z_k, similar_source_index, source_data)

        current_num_sources = len(similar_source_index)
        offset = p * current_num_sources
        M_current = np.array([idx - offset for idx in Az if idx >= offset])
        R_min = min(R_final, R_CoRT)
        R_min = min(R_min, z_max)
        R_min = max(z_k, R_min)

        if np.array_equal(M_current, M_obs) == True:
            z_interval.append([z_k, R_min])
        # print(z_k)
        z_k = R_min + 1e-5
    return z_interval

    

