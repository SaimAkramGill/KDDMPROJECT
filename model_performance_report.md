# Model Performance Report

## Model Comparison

| Model             |   Validation MAE |   Validation R² | CV MAE          |   Training MAE |
|:------------------|-----------------:|----------------:|:----------------|---------------:|
| Random Forest     |           0.4894 |         -0.0354 | 0.3882 ± 0.0978 |         0.2969 |
| Gradient Boosting |           0.4381 |         -0.02   | 0.3509 ± 0.1191 |         0.2956 |
| Linear Regression |           0.4846 |          0.0141 | 0.4073 ± 0.1092 |         0.3829 |
| Ridge Regression  |           0.465  |          0.0078 | 0.3743 ± 0.1207 |         0.3604 |
| Lasso Regression  |           0.4259 |         -0.0027 | 0.3298 ± 0.1210 |         0.3265 |

## Best Model: Lasso Regression

### Performance Metrics
- **Validation MAE**: 0.4259
- **Validation RMSE**: 1.8169
- **Validation R²**: -0.0027
- **Validation MAPE**: 39.25%
- **Cross-Validation MAE**: 0.3298 ± 0.1210

### Best Hyperparameters
- **alpha**: 1.0

## Model Insights

