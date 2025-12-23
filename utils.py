import numpy as np
from sklearn.linear_model import Lasso
import math
from sklearn.metrics import mean_squared_error
from mpmath import mp

def split_target(T, X_target, y_target, n_target):
  folds = []
  fold_size = math.floor(n_target / T)

  for i in range(T):
    start = i * fold_size
    if i == T-1:
      end = n_target
    else:
      end =  (i + 1) * fold_size

    X_fold = X_target[start:end]
    y_fold = y_target[start:end]

    folds.append({"X": X_fold, "y": y_fold})
  return folds

def get_active_X(model_coef, X):
  active_cols = [idx for idx, coef in enumerate(model_coef) if coef != 0]
  inactive_cols = [idx for idx, coef in enumerate(model_coef) if coef == 0]

  return X[:, active_cols], X[:, inactive_cols]

def construct_test_statistic(y, j, X_active):
  '''
  Compute test statistic direction etaj and its projection etajTy
  '''

  n, m = X_active.shape
  ej = np.zeros((m,1))
  ej[j] = 1
  etajT = ej.T @ np.linalg.pinv(X_active.T @ X_active + 1e-6 * np.eye(m)) @ X_active.T 
  etaj = etajT.T
  etajTy = etajT @ y

  return etaj, etajTy.item()

def get_affine_params(X_fold, y_fold_indices, a_global, b_global, source_data_k=None):
  X_out = X_fold
  a_out = a_global[y_fold_indices].ravel()
  b_out = b_global[y_fold_indices].ravel()

  if source_data_k is not None:
    X_source = source_data_k["X"]
    y_source = source_data_k["y"].ravel()
    n_source = len(y_source)

    X_out = np.vstack([X_out, X_source])
    a_out = np.hstack([a_out, y_source])
    b_zero = np.zeros(n_source)
    b_out = np.hstack([b_out, b_zero])

  return X_out, a_out, b_out

def computed_truncated_cdf(L, R, z, mu, sigma):
  """
  Computes the Truncated Normal CDF using high-precision arithmetic.
  """

  norm_L = (L - mu) / sigma
  norm_R = (R - mu) / sigma
  norm_z = (z - mu) / sigma

  cdf_L = mp.ncdf(norm_L)
  cdf_R = mp.ncdf(norm_R)
  cdf_y = mp.ncdf(norm_z)

  denominator = cdf_R - cdf_L

  if denominator == 0:
    return None
  
  numerator = cdf_y - cdf_L

  if numerator < 0:
    print("numerator is negative")

  if numerator > denominator:
    print("numerator is bigger than denominator")

  val = numerator / denominator

  return float(val)



def find_similar_source(z, a_global, b_global, alpha, n_target, K, target_data, source_data, T, verbose=False):
   
    X_target = target_data["X"]
    y_target = a_global + b_global * z

    similar_source_index = []
    threshold = (T + 1) / 2

    folds = split_target(T, X_target, y_target, n_target)

    for k in range(K):
        source_k = source_data[k]
        X_source_k = source_k["X"]
        y_source_k = source_k["y"].ravel()

        count = 0

        for t in range(T):
            X_test = folds[t]["X"]
            y_test = folds[t]["y"].ravel()

            X_train_list = [folds[i]["X"] for i in range(T) if i != t]
            y_train_list = [folds[i]["y"] for i in range(T) if i != t]

            X_train = np.vstack(X_train_list)
            y_train = np.concatenate(y_train_list).ravel()

           
            model_0 = Lasso(alpha, fit_intercept=False, tol=1e-10, max_iter=10000000)
            model_0.fit(X_train, y_train)

            pred_0 = model_0.predict(X_test)
            X_train_0k = np.vstack([X_train, X_source_k])
            y_train_0k = np.concatenate([y_train, y_source_k])

            
            model_0k = Lasso(alpha, fit_intercept=False, tol=1e-10, max_iter=10000000)
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
    
def pivot(A, list_active_set, list_zk, etaj, etajTy, tn_mu, cov):
    z_interval = []
    for i in range(len(list_active_set)):
        if np.array_equal(A, list_active_set[i]):
                z_interval.append([list_zk[i], list_zk[i+1] - 1e-10])

    # Merge intervals
    new_z_interval = []
    for interval in z_interval:
        if len(new_z_interval) == 0:
            new_z_interval.append(interval)
        else:
            dif = abs(interval[0] - new_z_interval[-1][1])
            if dif < 0.001:
                new_z_interval[-1][1] = interval[1]
            else:
                new_z_interval.append(interval)
    z_interval = new_z_interval

    tn_sigma = (np.sqrt(etaj.T @ cov @ etaj)).item()

    num = 0
    den = 0

    for interval in z_interval:
        lower = interval[0]
        upper = interval[1]

        # Normalize to standard normal Z = (x - mu) / sigma
        z_u = (upper - tn_mu) / tn_sigma
        z_l = (lower - tn_mu) / tn_sigma

        den += mp.ncdf(z_u) - mp.ncdf(z_l)

        if etajTy >= upper:
            num += mp.ncdf(z_u) - mp.ncdf(z_l)

        elif lower <= etajTy < upper:
            z_norm = (etajTy - tn_mu) / tn_sigma
            num += mp.ncdf(z_norm) - mp.ncdf(z_l)

    if den == 0:
        return None

    conditional_cdf = num / den
    p_value = 2 * min(conditional_cdf, 1 - conditional_cdf)
    
    return float(p_value)