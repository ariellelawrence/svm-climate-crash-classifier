# SVM Kernel Evaluation for Climate Model Simulation Crash Classification

Multi-kernel SVM evaluation and hyperparameter tuning for classifying climate model simulation crashes using the [UCI Climate Model Simulation Crashes dataset](https://archive.ics.uci.edu/ml/datasets/climate+model+simulation+crashes), which contains 540 simulation runs described by 18 physical parameters.

## Overview

This project compares four SVM kernels across a full ML pipeline — data cleaning, exploratory analysis, feature scaling, kernel evaluation, and hyperparameter tuning via GridSearchCV:

- **Linear** — baseline linear decision boundary
- **Polynomial** — higher-order feature interactions
- **RBF** — radial basis function, tuned via GridSearchCV
- **Sigmoid** — neural network-inspired kernel

## Results

| Model | Accuracy |
|---|---|
| **Linear SVM** | **0.961** |
| Tuned RBF SVM | 0.944 |
| RBF SVM | 0.927 |
| Polynomial SVM | 0.916 |
| Sigmoid SVM | 0.849 |

The linear kernel achieved the highest accuracy (96.1%), suggesting the crash/non-crash boundary is largely linearly separable. The tuned RBF model achieved perfect recall (1.00) — no simulation crashes missed — making it the preferred choice in contexts where missing a crash is costly.

## Tech Stack

Python · pandas · scikit-learn · matplotlib · seaborn
