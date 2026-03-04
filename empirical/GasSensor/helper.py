import os
import pandas as pd

from sklearn.datasets import load_svmlight_file
from sklearn.preprocessing import StandardScaler
import numpy as np

def get_target_data(df, n_target):
    target_df = df[df["batch_id"] == 10]
    
    n_samples = min(len(target_df), n_target)
    target_df = target_df.sample(n=n_samples)

    X = target_df.drop(columns=["target_y", "batch_id"])
    y = target_df["target_y"]

    X_scaler = StandardScaler()
    y_scaler = StandardScaler()

    X_scaled = X_scaler.fit_transform(X)
    y_scaled = y_scaler.fit_transform(y.values.reshape(-1, 1)).ravel()

    return {
        "X": X_scaled,
        "y": y_scaled
    }

def get_source_data(df, n_source):
    source_batch_ids = [5,6, 7, 8, 9]
    source_data = []

    for b_id in source_batch_ids:
        source_df = df[df["batch_id"] == b_id]
        
        n_samples = min(len(source_df), n_source)
        source_df = source_df.sample(n=n_samples)
        
        X = source_df.drop(columns=["target_y", "batch_id"])
        y = source_df["target_y"]

        X_scaler = StandardScaler()
        y_scaler = StandardScaler()

        X_scaled = X_scaler.fit_transform(X)
        y_scaled = y_scaler.fit_transform(y.values.reshape(-1, 1)).ravel()

        source_data.append({
            "X": X_scaled,
            "y": y_scaled
        })

    return source_data

def estimate_Sigma(X, y, n):
    from sklearn.linear_model import LassoCV

    clf = LassoCV(cv=10, fit_intercept=False).fit(X, y)

    beta_hat = clf.coef_
    p_lambda = beta_hat[beta_hat != 0].shape[0]
    res = y - X @ beta_hat

    if p_lambda >= n:
        raise ValueError("p_lambda must be less than n to estimate Sigma.")

    sigma_squared = (1 / (n - p_lambda)) * np.sum(res ** 2)

    Sigma = sigma_squared * np.eye(n)
    return Sigma

def load_local_gas_drift(folder_path='Dataset'):
    all_batches = []
    
    for i in range(1, 11):
        file_path = os.path.join(folder_path, f"batch{i}.dat")
        
        if os.path.exists(file_path):
            X_sparse, y = load_svmlight_file(file_path)
            
            df_batch = pd.DataFrame(X_sparse.toarray())

            df_batch['target_y'] = y
            df_batch['batch_id'] = i  
            
            all_batches.append(df_batch)
            print(f"Loaded {file_path}: {df_batch.shape}")
        else:
            print(f"Warning: {file_path} not found.")

    if not all_batches:
        return None

    full_df = pd.concat(all_batches, axis=0).reset_index(drop=True)
    
    # Rename features to feat_0, feat_1, ... feat_127
    feature_cols = [c for c in full_df.columns if isinstance(c, int)]
    full_df = full_df.rename(columns={c: f"feat_{c}" for c in feature_cols})

    return full_df