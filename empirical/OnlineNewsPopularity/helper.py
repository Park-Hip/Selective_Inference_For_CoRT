from sklearn.preprocessing import StandardScaler
import numpy as np

def get_target_data(df, n_target):
    target_df = df[df[" data_channel_is_entertainment"] == 1]
    target_df = target_df.sample(n=n_target)

    X = target_df.drop(columns=[" shares"])
    y = target_df[" shares"]

    X_scaler = StandardScaler()
    y_scaler = StandardScaler()

    X_scaled = X_scaler.fit_transform(X)
    y_scaled = y_scaler.fit_transform(y.values.reshape(-1, 1)).ravel()

    target_data = {
        "X": X_scaled,
        "y": y_scaled
    }
    return target_data

def get_source_data(df, n_source):
    columns = [" data_channel_is_lifestyle", " data_channel_is_bus", " data_channel_is_socmed", " data_channel_is_tech", " data_channel_is_world"]

    source_data = []
    for c in columns:
        source_df = df[df[c] == 1]
        source_df = source_df.sample(n=n_source)
        X = source_df.drop(columns=[" shares"])
        y= source_df[" shares"]

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