# CORT-SI: Selective Inference for CoRT

This repository provides an implementation of **Selective Inference** for CoRT. It includes algorithms for Over-conditioning and Parametric Programming.

## Project Structure

```text
cort_si/
├── CoRT_Builder.py                 # Core CoRT logic and synthetic data generation
├── oc.py                           # Implementation of Over-conditioning (OC) inference logic
├── parametric.py                   # Implementation of Parametric Programming logic
│
├── test/
│   ├── test_over_conditioning.ipynb        # Validates OC method (Uniform distribution check & TPR/FPR)
│   ├── test_original_parametric.ipynb      # Validates unoptimized parametric method (Uniform distribution check)
│   ├── test_optimized_parametric.ipynb     # Validates optimized parametric method (Uniform distribution check & TPR/FPR)
│   └── test_results_optim_and_origin.ipynb # Consistency check: compares outputs of optimized vs. unoptimized logic
│
└── reference/
    └── hdnl1.pdf                   # Mathematical foundation and derivations for the algorithms