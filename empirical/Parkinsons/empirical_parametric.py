import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

import pandas as pd
import numpy as np
import json
from helper import get_target_data, get_source_data
from methods import *
from configs import CONST_C
from ucimlrepo import fetch_ucirepo 

parkinsons_telemonitoring = fetch_ucirepo(id=189) 

X = parkinsons_telemonitoring.data.features.copy()
y = parkinsons_telemonitoring.data.targets 

subject_ids = parkinsons_telemonitoring.data.ids['subject#']
X["id"] = subject_ids 

df = pd.concat([X, y], axis=1)

n_target = 50
n_source = 100
iteration = 1000
p = df.shape[1] - 3
K = 5
T = 5
beta_index_list = [1, 5, 15]

method_name = "SI_parametric"
results = {
    method_name: {b: [] for b in beta_index_list}
}

def run():
    results = {
        method_name: {b: [] for b in beta_index_list}
    }  
    for i in range(iteration):
        
        target_data = get_target_data(df, n_target)
        source_data = get_source_data(df, n_source)

        print(f"Running {method_name} for iteration {i+1}/{iteration}...")
        result = SI_parametric_empirical(p, K, T, CONST_C, beta_index_list, target_data, source_data)

        for idx, b in enumerate(beta_index_list):
            if result[idx] is not None:
                results[method_name][b].append(result[idx])

        if i % 100 == 0:
            records_path = os.path.join(os.path.dirname(__file__), "records.json")
            if os.path.exists(records_path) and os.path.getsize(records_path) > 0:
                with open(records_path, "r") as f:
                    records = json.load(f)
            else:
                records = {}

            if method_name not in records:
                records[method_name] = {str(b): [] for b in beta_index_list}

            for b in beta_index_list:
                records[method_name][str(b)].extend(results[method_name][b])

            with open(records_path, "w") as f:
                json.dump(records, f, indent=2)

            results = {
                method_name: {b: [] for b in beta_index_list}
            }

            print(f"Results successfully saved to {records_path}")
    

if __name__ == "__main__":
    run()