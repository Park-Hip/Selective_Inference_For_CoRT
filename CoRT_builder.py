import numpy as np
from sklearn.linear_model import LassoCV, Lasso
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import utils

class CoRT:
    def __init__(self, use_LassoCV=False, alpha=0.1):
        self.use_LassoCV = use_LassoCV;
        self.alpha = alpha

    def gen_data(self, n_target, n_source, p, K, Ka, h, s_vector, s, cov_type):
        """
        Generate source and target data

        Args:
          - n_target: Number of target samples
          - n_source:  Number of source samples
          - p: Total number of features
          - K: Number of source tasks
          - Ka: Number of "similar" source tasks
          - h: Perturbation factor for source weights
          - s_vector: non-zero coefficients for first s features
          - s: number of non-zero coefficient
          - cov_type: Covariance type of the data. This can be 'standard' which is an identity matrix or 'AR' which stands for Auto regressor where:  Σij​ = σ^∣i−j∣

        Return:
          - target_data: dict
          - source_data: List[dict]
        """
        if not isinstance(s_vector, np.ndarray):
            s_vector = np.array(s_vector)

        # 1. Covariance Setup
        if cov_type == 'standard':
            sigma_val = 1.0
            Sigma = sigma = np.eye(p)
        elif cov_type == "AR":
            indices = np.arange(p)
            sigma_val = 0.5
            Sigma = sigma_val ** np.abs(indices[:, None] - indices[None, :])

        # 2. Target Data
        beta = np.concatenate([s_vector, np.zeros((p - s))]).reshape(-1, 1)

        X_target = np.random.multivariate_normal(mean=np.zeros(p), cov=Sigma, size=n_target)
        y_target = X_target @ beta + np.random.randn(n_target, 1)
        target_data = {"X": X_target, "y": y_target}

        # 3. Source Data
        source_data = []
        for k in range(K):
            eps = np.random.normal(0, 0.3, size=(p, 1))
            Sigma_source = Sigma + (eps @ eps.T)

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
                beta_k = beta_k + (2 * h / p) * np.random.choice([-1, 1], size=(p, 1)) ################
                ## beta_k = beta_k + (10 * h / p) * np.random.choice([-1, 1], size=(p, 1))
                y_k = X_k @ beta_k + np.random.randn(n_source, 1) + 0.5

            source_data.append({"X": X_k, "y": y_k})

        return target_data, source_data

    def find_similar_source(self, n_target, K, target_data, source_data, T=5, verbose=False):
        """
        Find similar sources using Adaptive Co-Regularization Detection (Algorithm 1)

        Args:
          - K: Number of source tasks
          - n_target: Number of target samples
          - T: split target_source into T (an odd number) parts
          - target_data: data for our target task
          - source_data: auxiliary data for our target

        Return:
          - similar_source_index: List
        """
        X_target = target_data["X"]
        y_target = target_data["y"]

        # X_target = X_target - X_target.mean(axis=0) #########
        # y_target = y_target - y_target.mean() #######

        similar_source_index = []
        threshold = (T + 1) / 2

        folds = utils.split_target(T, X_target, y_target, n_target)

        for k in range(K):
            source_k = source_data[k]
            X_source_k = source_k["X"]
            y_source_k = source_k["y"].ravel()

            # X_source_k = X_source_k - X_source_k.mean(axis=0) #############
            # y_source_k = y_source_k - y_source_k.mean() #################
            count = 0

            for t in range(T):
                X_test = folds[t]["X"]
                y_test = folds[t]["y"].ravel()

                X_train_list = [folds[i]["X"] for i in range(T) if i != t]
                y_train_list = [folds[i]["y"] for i in range(T) if i != t]

                X_train = np.vstack(X_train_list)
                y_train = np.concatenate(y_train_list).ravel()

                if self.use_LassoCV:
                    model_0 = LassoCV(cv=5, fit_intercept=False, random_state=42, n_jobs=-1)
                else:
                    model_0 = Lasso(alpha=self.alpha, fit_intercept=False, random_state=42)

                model_0.fit(X_train, y_train)
                pred_0 = model_0.predict(X_test)

                X_train_0k = np.vstack([X_train, X_source_k])
                y_train_0k = np.concatenate([y_train, y_source_k])

                if self.use_LassoCV:
                    model_0k = LassoCV(cv=5, fit_intercept=False, random_state=42, n_jobs=-1)
                else:
                    model_0k = Lasso(alpha=self.alpha, fit_intercept=False, random_state=42)

                model_0k.fit(X_train_0k, y_train_0k)
                pred_0k = model_0k.predict(X_test)

                loss_0 = mean_squared_error(y_test, pred_0)
                loss_0k = mean_squared_error(y_test, pred_0k)

                if loss_0k <= loss_0:
                    count += 1

            if count >= threshold:
                similar_source_index.append(k)

            if verbose:
                print(f"Total {len(similar_source_index)} similar sources: {similar_source_index}")

            return similar_source_index

    def prepare_CoRT_data(self, similar_source_index, source_data, target_data):
        """
        Prepare data for CoRT (Co-Regularization Transfer).

        Args:
          - similar_source_index: sources that are similar to the target, got via find_similar_source()
          - source_data: auxiliary data for our target
          - target_data: data for our target task

        Return:
          - X_combined: X matrix for CorT. Suppose we have K = 3 sources : S1, S2, S3 and X_T is the X_target.
                X_combined =  [ X_S1   0      0      X_S1 ]  # S1 rows
                              [ 0     X_S2    0      X_S2 ]  # S2 rows
                              [ 0      0     X_S3    X_S3 ]  # S3 rows
                              [ 0      0      0      X_T  ]  # Target rows
          - y_combined: Combined y of sources and target
        """
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