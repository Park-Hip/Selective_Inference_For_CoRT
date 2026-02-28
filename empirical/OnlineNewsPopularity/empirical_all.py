import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import json
from helper import get_target_data, get_source_data, estimate_Sigma
from methods import *

data_filename = os.path.join(os.path.dirname(__file__), "OnlineNewsPopularity.csv")
df = pd.read_csv(data_filename)
df = df.drop(columns=["url", " timedelta"])

n_target = 50
n_source = 100
iteration = 100
p = df.shape[1] - 1
K = 5
T = 5
beta_index_list = [5, 24, 25, 26, 28]


results = {
    "SI_parametric": {b: [] for b in beta_index_list},
    "SI_over_conditioning": {b: [] for b in beta_index_list},
    "data_splitting": {b: [] for b in beta_index_list},
    "bonferroni": {b: [] for b in beta_index_list}
}

methods = {
    "SI_parametric": SI_parametric_empirical,
    "SI_over_conditioning": SI_over_conditioning_empirical,
    "data_splitting": data_splitting_empirical,
    "bonferroni": bonferroni_empirical
}

def run():
    for i in range(iteration):
        target_data = get_target_data(df, n_target)
        source_data = get_source_data(df, n_source)

        for method_name, method_func in methods.items():
            print(f"Running {method_name} for iteration {i+1}/{iteration}...")
            if method_name in ["SI_parametric", "SI_over_conditioning"]:
                CONST_C = 25
            else:
                CONST_C = 1.2
            result = method_func(p, K, T, CONST_C, beta_index_list, target_data, source_data)
            if result is not None:
                for idx, b in enumerate(beta_index_list):
                    results[method_name][b].append(result[idx])

    records_path = os.path.join(os.path.dirname(__file__), "records.json")
    records = {method: {str(b): [] for b in beta_index_list} for method in methods}

    if os.path.exists(records_path) and os.path.getsize(records_path) > 0:
        try:
            with open(records_path, "r") as f:
                loaded_records = json.load(f)
                if isinstance(loaded_records, dict):
                    for method in records:
                        if method in loaded_records and isinstance(loaded_records[method], dict):
                            for b_str in records[method]:
                                if b_str in loaded_records[method]:
                                    records[method][b_str] = loaded_records[method][b_str]
        except (json.JSONDecodeError, ValueError):
            pass

    for method in records:
        for b in beta_index_list:
            records[method][str(b)].extend(results[method][b])

    with open(records_path, "w") as f:
        json.dump(records, f, indent=4)
    print(f"Results successfully saved to {records_path}")

if __name__ == "__main__":
    run()