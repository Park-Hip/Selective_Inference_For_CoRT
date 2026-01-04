import numpy as np
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error
from . import utils 

class CoRT:
    def __init__(self, alpha):
        self.alpha = alpha

    def gen_data(self, n_target, n_source, p, K, Ka, h, s_vector, s, cov_type):
        if not isinstance(s_vector, np.ndarray):
            s_vector = np.array(s_vector)

        if cov_type == 'standard':
            sigma_val = 1.0
            Sigma = np.eye(p)
            Sigma_source = Sigma

        elif cov_type == "AR":
            indices = np.arange(p)
            sigma_val = 0.5
            Sigma = sigma_val ** np.abs(indices[:, None] - indices[None, :])
            eps = np.random.normal(0, 0.3, size=(p, 1))
            Sigma_source = Sigma + (eps @ eps.T)

        beta = np.concatenate([s_vector, np.zeros((p - s))]).reshape(-1, 1)

        X_target = np.random.multivariate_normal(mean=np.zeros(p), cov=Sigma, size=n_target)
        y_target = X_target @ beta + np.random.randn(n_target, 1)
        target_data = {"X": X_target, "y": y_target}

        source_data = []
        for k in range(K):
            X_k = np.random.multivariate_normal(mean=np.zeros(p), cov=Sigma_source, size=n_source)

            if k < Ka:
                perpetuation = np.zeros((p, 1))
                perpetuation[:s] = (h / p) * np.random.choice([-1, 1], size=(s, 1))
                beta_k = beta + perpetuation
                y_k = X_k @ beta_k + np.random.randn(n_source, 1)
            else:
                beta_k = np.zeros((p, 1))
                idx_shift = np.arange(s, 2 * s)
                idx_random = np.random.choice(np.arange(2 * s, p), size=s, replace=False)
                active_indices = np.concatenate([idx_shift, idx_random])
                beta_k[active_indices] = 0.5
                beta_k = beta_k + (2 * h / p) * np.random.choice([-1, 1], size=(p, 1)) 
                y_k = X_k @ beta_k + np.random.randn(n_source, 1) + 0.5

            source_data.append({"X": X_k, "y": y_k})
        return target_data, source_data

    def find_similar_source(self, n_target, K, target_data, source_data, lamda_not_source, lamda_1_source, T, verbose=False):
        X_target = target_data["X"]
        y_target = target_data["y"]

        similar_source_index = []
        threshold = (T + 1) / 2
        folds = utils.split_target(T, X_target, y_target, n_target)
        cnt = {}
        # print(self.alpha)
        for t in range(T):
            X_test = folds[t]["X"]
            y_test = folds[t]["y"].ravel()
            
            X_train_list = [folds[i]["X"] for i in range(T) if i != t]
            y_train_list = [folds[i]["y"] for i in range(T) if i != t]
            X_train = np.vstack(X_train_list)
            y_train = np.concatenate(y_train_list).ravel()
            
            model_0 = Lasso(alpha=lamda_not_source, fit_intercept=False, tol=1e-10, max_iter=1000000)
            model_0.fit(X_train, y_train)
            pred_0 = model_0.predict(X_test)
            loss_0 = mean_squared_error(y_test, pred_0)
            
            for k in range(K):
                source_k = source_data[k]
                X_source_k = source_k["X"]
                y_source_k = source_k["y"].ravel()
                
                X_train_0k = np.vstack([X_train, X_source_k])
                y_train_0k = np.concatenate([y_train, y_source_k])

                model_0k = Lasso(alpha=lamda_1_source, fit_intercept=False, tol=1e-10, max_iter=1000000)
                model_0k.fit(X_train_0k, y_train_0k)
                pred_0k = model_0k.predict(X_test)

                loss_0k = mean_squared_error(y_test, pred_0k)

                if loss_0k <= loss_0:
                    cnt[(k, t)] = 1
                else:
                    cnt[(k, t)] = 0
        
        for k in range(K):
            count = 0
            for t in range(T):
                if cnt[(k, t)] == 1:
                    count += 1
            if count >= threshold:
                similar_source_index.append(k)

        if verbose:
            print(f"Total {len(similar_source_index)} similar sources: {similar_source_index}")

        return similar_source_index

    def prepare_CoRT_data(self, similar_source_index, source_data, target_data):
        X_target = target_data["X"]
        y_target = target_data["y"].reshape(-1, 1)
        p = X_target.shape[1]

        similar_source_data = [source_data[i] for i in similar_source_index]
        similar_source_count = len(similar_source_data)

        total_cols = p * (similar_source_count + 1)
        X_blocks = []
        y_combined = []

        for i, data in enumerate(similar_source_data):
            X_k = data["X"]
            y_k = data["y"].reshape(-1, 1)

            left_cols = i * p
            right_cols = total_cols - left_cols - 2 * p
            X_block = np.hstack([
                np.zeros((X_k.shape[0], left_cols)),
                X_k,
                np.zeros((X_k.shape[0], right_cols)),
                X_k
            ])
            X_blocks.append(X_block)
            y_combined.append(y_k)

        X_target_block = np.hstack([
            np.zeros((X_target.shape[0], p * similar_source_count)),
            X_target
        ])
        X_blocks.append(X_target_block)
        y_combined.append(y_target)

        X_combined = np.vstack(X_blocks)
        y_combined = np.vstack(y_combined)

        return X_combined, y_combined