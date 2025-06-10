# Data Preprocessing Report

## Data Quality Analysis
- **Original Shape**: 1,041 rows × 23 columns
- **High Missing Features (>30%)**: 2
- **Target Missing**: 5
- **Target Outliers**: 11

## Data Cleaning
- **Rows Removed**: 50 (4.80%)
- **Final Dataset**: 991 rows

## Missing Value Imputation
- **Numerical Features**: Median imputation
- **Categorical Features**: Mode imputation

## Outlier Handling
- **Total Outliers Handled**: 667
- **Features Processed**: 6
- **Method**: IQR-based capping

## Categorical Encoding
- **Features Encoded**: 10
- **Method**: Label Encoding

## Feature Scaling
- **Features Scaled**: 11
- **Method**: MINMAX

## Feature Selection
- **Available Features**: 21
- **Selected Features**: 15
- **Selection Method**: Correlation

## Data Splitting
- **Training**: 594 (59.9%)
- **Validation**: 198 (20.0%)
- **Test**: 199 (20.1%)
