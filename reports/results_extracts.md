# Results extracts

## Table 1 (survey-weighted, afebrile training sample)

| country | n | rdt_pos | age_mean | pct_female | pct_urban | pct_net_used | wealth_mean | weighted_rdt_prev |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Burkina Faso | 15309 | 6613 | 33.12 | 48.85 | 19.59 | 63.11 | 2.903 | 43.08 |
| Gambia | 5001 | 46 | 31.48 | 47.32 | 55.77 | 48.65 | 2.879 | 1.275 |
| Mali | 17033 | 3922 | 32.62 | 48.91 | 19.42 | 76.59 | 2.914 | 25.09 |
| Mauritania | 7063 | 54 | 32.69 | 48.69 | 41.63 | 20.64 | 2.817 | 0.9376 |
| Niger | 2737 | 410 | 32.2 | 49.41 | 19.54 | 86.31 | 3.038 | 19.5 |
| Nigeria | 15979 | 5195 | 32.9 | 48.56 | 40.02 | 45.21 | 3.108 | 32.25 |
| Senegal | 30074 | 573 | 32.82 | 49.09 | 36.48 | 56.25 | 2.778 | 1.257 |

## LOCO discrimination and calibration (primary LightGBM/elastic net)

| country | model | n | rdt_pos | auc | oe | slope | intercept | flag_pos_lt50 | params | tag | auc_ci_low | auc_ci_high | oe_ci_low | oe_ci_high |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Burkina Faso | elasticnet | 15309 | 6613 | 0.7082 | 2.128 | 0.7228 | 0.8858 | 0 | {"l1_ratio": 0.5, "C": 1.0} | primary | 0.6906 | 0.7239 | 2.049 | 2.213 |
| Burkina Faso | lightgbm | 15309 | 6613 | 0.7097 | 2.277 | 0.518 | 0.7398 | 0 | {"num_leaves": 31, "learning_rate": 0.05, "n_estimators": 300, "max_depth": 5} | primary | 0.6904 | 0.7254 | 2.171 | 2.383 |
| Gambia | elasticnet | 5001 | 46 | 0.5988 | 0.1606 | 0.362 | -3.396 | 1 | {"l1_ratio": 0.5, "C": 1.0} | primary | 0.51 | 0.6847 | 0.09356 | 0.2462 |
| Gambia | lightgbm | 5001 | 46 | 0.5579 | 0.1519 | 0.2654 | -3.688 | 1 | {"num_leaves": 15, "learning_rate": 0.05, "n_estimators": 200, "max_depth": 3} | primary | 0.4743 | 0.6515 | 0.08881 | 0.2323 |
| Mali | elasticnet | 17033 | 3922 | 0.6682 | 0.9236 | 0.4939 | -0.5359 | 0 | {"l1_ratio": 0.5, "C": 1.0} | primary | 0.6435 | 0.6934 | 0.8575 | 0.9932 |
| Mali | lightgbm | 17033 | 3922 | 0.6757 | 0.9548 | 0.498 | -0.4981 | 0 | {"num_leaves": 15, "learning_rate": 0.05, "n_estimators": 200, "max_depth": 3} | primary | 0.651 | 0.7021 | 0.8909 | 1.025 |
| Mauritania | elasticnet | 7063 | 54 | 0.5807 | 0.2164 | 0.2004 | -3.941 | 0 | {"l1_ratio": 0.5, "C": 1.0} | primary | 0.4562 | 0.7067 | 0.1492 | 0.2905 |
| Mauritania | lightgbm | 7063 | 54 | 0.5574 | 0.1257 | 0.11 | -4.274 | 0 | {"num_leaves": 31, "learning_rate": 0.1, "n_estimators": 150, "max_depth": 4} | primary | 0.4491 | 0.6793 | 0.08708 | 0.1709 |
| Niger | elasticnet | 2737 | 410 | 0.6618 | 0.7881 | 0.5073 | -0.7972 | 0 | {"l1_ratio": 0.5, "C": 1.0} | primary | 0.6132 | 0.7097 | 0.6625 | 0.9305 |
| Niger | lightgbm | 2737 | 410 | 0.6254 | 0.8134 | 0.3892 | -0.8983 | 0 | {"num_leaves": 31, "learning_rate": 0.1, "n_estimators": 150, "max_depth": 4} | primary | 0.5722 | 0.6753 | 0.6708 | 0.9751 |
| Nigeria | elasticnet | 15979 | 5195 | 0.6901 | 1.213 | 0.6068 | -0.03668 | 0 | {"l1_ratio": 0.5, "C": 1.0} | primary | 0.6743 | 0.7051 | 1.162 | 1.266 |
| Nigeria | lightgbm | 15979 | 5195 | 0.7133 | 1.454 | 0.6775 | 0.2376 | 0 | {"num_leaves": 15, "learning_rate": 0.05, "n_estimators": 200, "max_depth": 3} | primary | 0.6984 | 0.7272 | 1.39 | 1.528 |
| Senegal | elasticnet | 30074 | 573 | 0.7755 | 0.132 | 0.6789 | -2.791 | 0 | {"l1_ratio": 0.5, "C": 1.0} | primary | 0.7332 | 0.8107 | 0.1104 | 0.1563 |
| Senegal | lightgbm | 30074 | 573 | 0.665 | 0.0831 | 0.3177 | -3.685 | 0 | {"num_leaves": 31, "learning_rate": 0.05, "n_estimators": 300, "max_depth": 5} | primary | 0.6199 | 0.7086 | 0.06916 | 0.09818 |

Calibration plots: `outputs/figures/figure1_calibration.png`.
DCA (fever rule vs fever+model on all 6–59 months with RDT): `outputs/figures/figure2_dca.png`.
SHAP is a statement about model behavior, not malaria biology.