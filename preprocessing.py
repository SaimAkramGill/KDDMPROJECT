"""
Data Preprocessing Module
Handles all data cleaning, transformation, and preparation for modeling
"""

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split
from typing import Dict, List, Tuple, Optional
import pickle
import warnings
warnings.filterwarnings('ignore')

class DataPreprocessor:
    """
    Class for comprehensive data preprocessing
    """
    
    def __init__(self, target_column: str = 'win_prob'):
        """
        Initialize preprocessor
        
        Args:
            target_column (str): Name of target variable
        """
        self.target_column = target_column
        self.numerical_features = [
            'power_level', 'weight', 'height', 'age', 'gender', 'speed', 
            'battle_iq', 'ranking', 'intelligence', 'training_time', 'secret_code'
        ]
        self.categorical_features = [
            'role', 'skin_type', 'eye_color', 'hair_color', 'universe', 
            'body_type', 'job', 'species', 'abilities', 'special_attack'
        ]
        
        # Preprocessing components
        self.numerical_imputer = None
        self.categorical_imputer = None
        self.scaler = None
        self.encoders = {}
        self.feature_selector = None
        
        # Processed data storage
        self.processed_data = None
        self.selected_features = []
        self.preprocessing_stats = {}
        
        # Original data backup
        self.original_data = None
    
    def analyze_data_quality(self, data: pd.DataFrame) -> Dict:
        """
        Analyze data quality issues
        
        Args:
            data (pd.DataFrame): Input dataset
            
        Returns:
            Dict: Data quality analysis results
        """
        print("=" * 60)
        print("DATA QUALITY ANALYSIS")
        print("=" * 60)
        
        analysis = {
            'total_rows': len(data),
            'total_columns': len(data.columns),
            'missing_values': {},
            'outliers': {},
            'data_types': {},
            'unique_values': {},
            'target_issues': {}
        }
        
        # Missing values analysis
        print("\n1. MISSING VALUES ANALYSIS")
        print("-" * 40)
        
        for column in data.columns:
            missing_count = data[column].isnull().sum()
            missing_pct = (missing_count / len(data)) * 100
            
            analysis['missing_values'][column] = {
                'count': missing_count,
                'percentage': missing_pct
            }
            
            if missing_count > 0:
                print(f"   {column}: {missing_count} ({missing_pct:.2f}%)")
        
        # Data types analysis
        print("\n2. DATA TYPES ANALYSIS")
        print("-" * 40)
        
        for column in data.columns:
            dtype = str(data[column].dtype)
            analysis['data_types'][column] = dtype
            
            if column in self.numerical_features:
                if not pd.api.types.is_numeric_dtype(data[column]):
                    print(f"     {column}: Expected numeric, got {dtype}")
        
        # Target variable analysis
        print("\n3. TARGET VARIABLE ANALYSIS")
        print("-" * 40)
        
        if self.target_column in data.columns:
            target_data = data[self.target_column].dropna()
            
            target_analysis = {
                'missing_count': data[self.target_column].isnull().sum(),
                'negative_values': (target_data < 0).sum(),
                'zero_values': (target_data == 0).sum(),
                'extreme_values': (target_data > target_data.quantile(0.99)).sum(),
                'infinite_values': np.isinf(target_data).sum()
            }
            
            analysis['target_issues'] = target_analysis
            
            print(f"   Missing values: {target_analysis['missing_count']}")
            print(f"   Negative values: {target_analysis['negative_values']}")
            print(f"   Zero values: {target_analysis['zero_values']}")
            print(f"   Extreme values (>99th percentile): {target_analysis['extreme_values']}")
            print(f"   Infinite values: {target_analysis['infinite_values']}")
        
        # Outliers analysis for numerical features
        print("\n4. OUTLIERS ANALYSIS")
        print("-" * 40)
        
        for feature in self.numerical_features:
            if feature in data.columns and pd.api.types.is_numeric_dtype(data[feature]):
                feature_data = data[feature].dropna()
                
                if len(feature_data) > 0:
                    q1 = feature_data.quantile(0.25)
                    q3 = feature_data.quantile(0.75)
                    iqr = q3 - q1
                    
                    lower_bound = q1 - 1.5 * iqr
                    upper_bound = q3 + 1.5 * iqr
                    
                    outliers = feature_data[(feature_data < lower_bound) | (feature_data > upper_bound)]
                    outlier_pct = len(outliers) / len(feature_data) * 100
                    
                    analysis['outliers'][feature] = {
                        'count': len(outliers),
                        'percentage': outlier_pct,
                        'lower_bound': lower_bound,
                        'upper_bound': upper_bound,
                        'min_value': feature_data.min(),
                        'max_value': feature_data.max()
                    }
                    
                    if outlier_pct > 0:
                        print(f"   {feature}: {len(outliers)} outliers ({outlier_pct:.2f}%)")
        
        # Unique values analysis for categorical features
        print("\n5. CATEGORICAL FEATURES ANALYSIS")
        print("-" * 40)
        
        for feature in self.categorical_features:
            if feature in data.columns:
                unique_count = data[feature].nunique()
                total_count = len(data[feature].dropna())
                
                analysis['unique_values'][feature] = {
                    'unique_count': unique_count,
                    'total_count': total_count,
                    'cardinality_ratio': unique_count / total_count if total_count > 0 else 0
                }
                
                print(f"   {feature}: {unique_count} unique values")
                
                if unique_count == total_count:
                    print(f"        High cardinality - each value is unique")
                elif unique_count == 1:
                    print(f"        No variation - all values are the same")
        
        self.preprocessing_stats['data_quality'] = analysis
        return analysis
    
    def clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Clean the dataset by removing problematic rows/columns
        
        Args:
            data (pd.DataFrame): Input dataset
            
        Returns:
            pd.DataFrame: Cleaned dataset
        """
        print("\n" + "=" * 60)
        print("DATA CLEANING")
        print("=" * 60)
        
        cleaned_data = data.copy()
        initial_rows = len(cleaned_data)
        
        # 1. Remove rows with missing target variable
        if self.target_column in cleaned_data.columns:
            target_missing = cleaned_data[self.target_column].isnull().sum()
            cleaned_data = cleaned_data.dropna(subset=[self.target_column])
            print(f"1. Removed {target_missing} rows with missing target variable")
        
        # 2. Remove duplicate rows
        duplicates = cleaned_data.duplicated().sum()
        if duplicates > 0:
            cleaned_data = cleaned_data.drop_duplicates()
            print(f"2. Removed {duplicates} duplicate rows")
        else:
            print("2. No duplicate rows found")
        
        # 3. Handle infinite values in numerical columns
        inf_values_fixed = 0
        for feature in self.numerical_features:
            if feature in cleaned_data.columns:
                inf_mask = np.isinf(cleaned_data[feature])
                if inf_mask.any():
                    # Replace infinite values with NaN
                    cleaned_data.loc[inf_mask, feature] = np.nan
                    inf_values_fixed += inf_mask.sum()
        
        if inf_values_fixed > 0:
            print(f"3. Converted {inf_values_fixed} infinite values to NaN")
        else:
            print("3. No infinite values found")
        
        # 4. Remove rows with too many missing values (>50% of features)
        missing_threshold = 0.5
        missing_per_row = cleaned_data.isnull().sum(axis=1) / len(cleaned_data.columns)
        rows_to_remove = missing_per_row > missing_threshold
        removed_rows = rows_to_remove.sum()
        
        if removed_rows > 0:
            cleaned_data = cleaned_data[~rows_to_remove]
            print(f"4. Removed {removed_rows} rows with >{missing_threshold*100}% missing values")
        else:
            print(f"4. No rows with excessive missing values (>{missing_threshold*100}%)")
        
        final_rows = len(cleaned_data)
        rows_removed = initial_rows - final_rows
        
        print(f"\n Cleaning Summary:")
        print(f"   Initial rows: {initial_rows}")
        print(f"   Final rows: {final_rows}")
        print(f"   Rows removed: {rows_removed} ({rows_removed/initial_rows*100:.2f}%)")
        
        self.preprocessing_stats['cleaning'] = {
            'initial_rows': initial_rows,
            'final_rows': final_rows,
            'rows_removed': rows_removed,
            'removal_percentage': rows_removed/initial_rows*100
        }
        
        return cleaned_data
    
    def handle_missing_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values through imputation
        
        Args:
            data (pd.DataFrame): Input dataset
            
        Returns:
            pd.DataFrame: Dataset with imputed values
        """
        print("\n" + "=" * 60)
        print("MISSING VALUES IMPUTATION")
        print("=" * 60)
        
        imputed_data = data.copy()
        
        # Filter features that exist in the dataset
        existing_numerical = [f for f in self.numerical_features if f in imputed_data.columns]
        existing_categorical = [f for f in self.categorical_features if f in imputed_data.columns]
        
        imputation_summary = {
            'numerical': {},
            'categorical': {}
        }
        
        # 1. Numerical features imputation
        if existing_numerical:
            print("1. NUMERICAL FEATURES IMPUTATION")
            print("-" * 40)
            
            # Choose imputation strategy based on data distribution
            for feature in existing_numerical:
                missing_count = imputed_data[feature].isnull().sum()
                if missing_count > 0:
                    feature_data = imputed_data[feature].dropna()
                    
                    # Check if data is highly skewed
                    skewness = feature_data.skew() if len(feature_data) > 0 else 0
                    
                    if abs(skewness) > 2:  # Highly skewed
                        strategy = 'median'
                    else:
                        strategy = 'mean'
                    
                    print(f"   {feature}: {missing_count} missing values (using {strategy})")
                    imputation_summary['numerical'][feature] = {
                        'missing_count': missing_count,
                        'strategy': strategy,
                        'skewness': skewness
                    }
            
            # Apply imputation
            self.numerical_imputer = SimpleImputer(strategy='median')  # Median is more robust
            imputed_data[existing_numerical] = self.numerical_imputer.fit_transform(
                imputed_data[existing_numerical]
            )
            
            print(f"    Imputed {len(existing_numerical)} numerical features using median strategy")
        
        # 2. Categorical features imputation
        if existing_categorical:
            print("\n2. CATEGORICAL FEATURES IMPUTATION")
            print("-" * 40)
            
            for feature in existing_categorical:
                missing_count = imputed_data[feature].isnull().sum()
                if missing_count > 0:
                    # Get the most frequent value
                    mode_value = imputed_data[feature].mode()
                    mode_value = mode_value[0] if len(mode_value) > 0 else 'Unknown'
                    
                    print(f"   {feature}: {missing_count} missing values (using mode: '{mode_value}')")
                    imputation_summary['categorical'][feature] = {
                        'missing_count': missing_count,
                        'mode_value': mode_value
                    }
            
            # Apply imputation
            self.categorical_imputer = SimpleImputer(strategy='most_frequent')
            imputed_data[existing_categorical] = self.categorical_imputer.fit_transform(
                imputed_data[existing_categorical]
            )
            
            print(f"    Imputed {len(existing_categorical)} categorical features using mode strategy")
        
        # Verify no missing values remain
        remaining_missing = imputed_data.isnull().sum().sum()
        print(f"\n Imputation Summary:")
        print(f"   Remaining missing values: {remaining_missing}")
        
        if remaining_missing > 0:
            print("     Some missing values remain in other columns")
        else:
            print("    All missing values handled successfully")
        
        self.preprocessing_stats['imputation'] = imputation_summary
        return imputed_data
    
    def handle_outliers(self, data: pd.DataFrame, method: str = 'iqr', action: str = 'cap') -> pd.DataFrame:
        """
        Handle outliers in numerical features
        
        Args:
            data (pd.DataFrame): Input dataset
            method (str): Outlier detection method ('iqr', 'zscore', 'percentile')
            action (str): Action to take ('cap', 'remove', 'transform')
            
        Returns:
            pd.DataFrame: Dataset with handled outliers
        """
        print("\n" + "=" * 60)
        print("OUTLIER HANDLING")
        print("=" * 60)
        
        outlier_data = data.copy()
        existing_numerical = [f for f in self.numerical_features if f in outlier_data.columns]
        
        outlier_summary = {}
        total_outliers_handled = 0
        
        print(f"Method: {method.upper()}, Action: {action.upper()}")
        print("-" * 40)
        
        for feature in existing_numerical:
            if feature == self.target_column:
                continue  # Don't modify target variable
            
            feature_data = outlier_data[feature].copy()
            initial_outliers = 0
            
            if method == 'iqr':
                q1 = feature_data.quantile(0.25)
                q3 = feature_data.quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                
                outlier_mask = (feature_data < lower_bound) | (feature_data > upper_bound)
                
            elif method == 'zscore':
                mean = feature_data.mean()
                std = feature_data.std()
                z_scores = np.abs((feature_data - mean) / std)
                outlier_mask = z_scores > 3  # 3 standard deviations
                
                lower_bound = mean - 3 * std
                upper_bound = mean + 3 * std
                
            elif method == 'percentile':
                lower_bound = feature_data.quantile(0.01)
                upper_bound = feature_data.quantile(0.99)
                outlier_mask = (feature_data < lower_bound) | (feature_data > upper_bound)
            
            initial_outliers = outlier_mask.sum()
            
            if initial_outliers > 0:
                if action == 'cap':
                    # Cap outliers to the bounds
                    outlier_data.loc[feature_data < lower_bound, feature] = lower_bound
                    outlier_data.loc[feature_data > upper_bound, feature] = upper_bound
                    action_taken = "capped"
                    
                elif action == 'remove':
                    # Remove rows with outliers (not recommended for this dataset)
                    outlier_data = outlier_data[~outlier_mask]
                    action_taken = "removed"
                    
                elif action == 'transform':
                    # Log transformation for positive skewed data
                    if feature_data.min() > 0:
                        outlier_data[feature] = np.log1p(feature_data)
                        action_taken = "log transformed"
                    else:
                        # Use IQR capping as fallback
                        outlier_data.loc[feature_data < lower_bound, feature] = lower_bound
                        outlier_data.loc[feature_data > upper_bound, feature] = upper_bound
                        action_taken = "capped (log transform not applicable)"
                
                print(f"   {feature}: {initial_outliers} outliers {action_taken}")
                total_outliers_handled += initial_outliers
                
                outlier_summary[feature] = {
                    'outliers_count': initial_outliers,
                    'action_taken': action_taken,
                    'lower_bound': lower_bound,
                    'upper_bound': upper_bound
                }
        
        print(f"\n Outlier Handling Summary:")
        print(f"   Total outliers handled: {total_outliers_handled}")
        print(f"   Features processed: {len(outlier_summary)}")
        
        self.preprocessing_stats['outliers'] = outlier_summary
        return outlier_data
    
    def encode_categorical_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Encode categorical features
        
        Args:
            data (pd.DataFrame): Input dataset
            
        Returns:
            pd.DataFrame: Dataset with encoded categorical features
        """
        print("\n" + "=" * 60)
        print("CATEGORICAL ENCODING")
        print("=" * 60)
        
        encoded_data = data.copy()
        existing_categorical = [f for f in self.categorical_features if f in encoded_data.columns]
        
        encoding_summary = {}
        
        for feature in existing_categorical:
            unique_values = encoded_data[feature].nunique()
            
            # Use Label Encoding for simplicity
            le = LabelEncoder()
            
            # Handle any remaining NaN values
            feature_data = encoded_data[feature].fillna('Unknown')
            
            encoded_data[f'{feature}_encoded'] = le.fit_transform(feature_data)
            self.encoders[feature] = le
            
            encoding_summary[feature] = {
                'unique_values': unique_values,
                'encoding_type': 'LabelEncoder',
                'encoded_column': f'{feature}_encoded'
            }
            
            print(f"   {feature}: {unique_values} unique values → {feature}_encoded")
        
        print(f"\n Encoding Summary:")
        print(f"   Categorical features encoded: {len(existing_categorical)}")
        print(f"   New encoded columns created: {len(existing_categorical)}")
        
        self.preprocessing_stats['encoding'] = encoding_summary
        return encoded_data
    
    def scale_features(self, data: pd.DataFrame, method: str = 'minmax') -> pd.DataFrame:
        """
        Scale numerical features
        
        Args:
            data (pd.DataFrame): Input dataset
            method (str): Scaling method ('minmax', 'standard', 'robust')
            
        Returns:
            pd.DataFrame: Dataset with scaled features
        """
        print("\n" + "=" * 60)
        print("FEATURE SCALING")
        print("=" * 60)
        
        scaled_data = data.copy()
        existing_numerical = [f for f in self.numerical_features if f in scaled_data.columns]
        
        # Choose scaler based on method
        if method == 'minmax':
            self.scaler = MinMaxScaler()
            scaler_name = "MinMaxScaler (0-1 range)"
        elif method == 'standard':
            self.scaler = StandardScaler()
            scaler_name = "StandardScaler (mean=0, std=1)"
        elif method == 'robust':
            self.scaler = RobustScaler()
            scaler_name = "RobustScaler (median=0, IQR=1)"
        else:
            raise ValueError(f"Unknown scaling method: {method}")
        
        print(f"Scaling method: {scaler_name}")
        print("-" * 40)
        
        if existing_numerical:
            # Apply scaling
            scaled_features = [f'{feature}_scaled' for feature in existing_numerical]
            scaled_data[scaled_features] = self.scaler.fit_transform(scaled_data[existing_numerical])
            
            # Print scaling summary
            for i, feature in enumerate(existing_numerical):
                original_range = f"[{scaled_data[feature].min():.2f}, {scaled_data[feature].max():.2f}]"
                scaled_range = f"[{scaled_data[scaled_features[i]].min():.2f}, {scaled_data[scaled_features[i]].max():.2f}]"
                print(f"   {feature}: {original_range} → {scaled_range}")
            
            print(f"\n Scaling Summary:")
            print(f"   Features scaled: {len(existing_numerical)}")
            print(f"   Scaling method: {method}")
            print(f"   New scaled columns: {len(scaled_features)}")
            
            self.preprocessing_stats['scaling'] = {
                'method': method,
                'features_scaled': existing_numerical,
                'scaled_columns': scaled_features
            }
        else:
            print("   No numerical features found to scale")
        
        return scaled_data
    
    def select_features(self, data: pd.DataFrame, method: str = 'correlation', top_k: int = 15) -> List[str]:
        """
        Select most important features for modeling
        
        Args:
            data (pd.DataFrame): Input dataset
            method (str): Feature selection method ('correlation', 'variance', 'all')
            top_k (int): Number of top features to select
            
        Returns:
            List[str]: Selected feature names
        """
        print("\n" + "=" * 60)
        print("FEATURE SELECTION")
        print("=" * 60)
        
        # Get all processed features
        scaled_features = [col for col in data.columns if col.endswith('_scaled')]
        encoded_features = [col for col in data.columns if col.endswith('_encoded')]
        all_features = scaled_features + encoded_features
        
        print(f"Available features for selection: {len(all_features)}")
        print(f"  - Scaled numerical: {len(scaled_features)}")
        print(f"  - Encoded categorical: {len(encoded_features)}")
        
        if method == 'correlation':
            print(f"\nMethod: Correlation-based selection (top {top_k})")
            print("-" * 40)
            
            # Calculate correlation with target
            correlations = {}
            for feature in all_features:
                if feature in data.columns:
                    corr = data[feature].corr(data[self.target_column])
                    if pd.notnull(corr):
                        correlations[feature] = abs(corr)  # Use absolute correlation
            
            # Sort by correlation and select top k
            sorted_features = sorted(correlations.items(), key=lambda x: x[1], reverse=True)
            
            # Apply minimum correlation threshold
            min_correlation = 0.02
            filtered_features = [(f, c) for f, c in sorted_features if c >= min_correlation]
            
            # Select top k features
            selected_features = [f for f, c in filtered_features[:top_k]]
            
            print("Top features by correlation:")
            for i, (feature, corr) in enumerate(filtered_features[:top_k]):
                print(f"   {i+1:2d}. {feature}: {corr:.4f}")
            
            if len(filtered_features) < top_k:
                print(f"   Note: Only {len(filtered_features)} features meet minimum correlation threshold ({min_correlation})")
        
        elif method == 'variance':
            print(f"\nMethod: Variance-based selection (top {top_k})")
            print("-" * 40)
            
            # Calculate variance for each feature
            variances = {}
            for feature in all_features:
                if feature in data.columns:
                    var = data[feature].var()
                    if pd.notnull(var) and var > 0:
                        variances[feature] = var
            
            # Sort by variance and select top k
            sorted_features = sorted(variances.items(), key=lambda x: x[1], reverse=True)
            selected_features = [f for f, v in sorted_features[:top_k]]
            
            print("Top features by variance:")
            for i, (feature, var) in enumerate(sorted_features[:top_k]):
                print(f"   {i+1:2d}. {feature}: {var:.4f}")
        
        elif method == 'all':
            print("\nMethod: Use all available features")
            print("-" * 40)
            selected_features = all_features
            print(f"   Selected all {len(selected_features)} features")
        
        else:
            raise ValueError(f"Unknown feature selection method: {method}")
        
        self.selected_features = selected_features
        
        print(f"\n Feature Selection Summary:")
        print(f"   Total available features: {len(all_features)}")
        print(f"   Selected features: {len(selected_features)}")
        print(f"   Selection method: {method}")
        
        self.preprocessing_stats['feature_selection'] = {
            'method': method,
            'total_features': len(all_features),
            'selected_features': len(selected_features),
            'selected_feature_names': selected_features
        }
        
        return selected_features
    
    def split_data(self, data: pd.DataFrame, test_size: float = 0.2, 
                   validation_size: float = 0.2, random_state: int = 42) -> Tuple:
        """
        Split data into train, validation, and test sets
        
        Args:
            data (pd.DataFrame): Preprocessed dataset
            test_size (float): Proportion of test set
            validation_size (float): Proportion of validation set (from remaining data)
            random_state (int): Random state for reproducibility
            
        Returns:
            Tuple: (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        print("\n" + "=" * 60)
        print("DATA SPLITTING")
        print("=" * 60)
        
        # Prepare features and target
        X = data[self.selected_features]
        y = data[self.target_column]
        
        print(f"Dataset shape: {X.shape}")
        print(f"Features used: {len(self.selected_features)}")
        print(f"Target variable: {self.target_column}")
        
        # First split: separate test set
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, shuffle=True
        )
        
        # Second split: separate train and validation from remaining data
        val_size_adjusted = validation_size / (1 - test_size)  # Adjust for remaining data
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted, random_state=random_state, shuffle=True
        )
        
        print(f"\n Data Split Summary:")
        print(f"   Training set: {X_train.shape[0]} samples ({X_train.shape[0]/len(X)*100:.1f}%)")
        print(f"   Validation set: {X_val.shape[0]} samples ({X_val.shape[0]/len(X)*100:.1f}%)")
        print(f"   Test set: {X_test.shape[0]} samples ({X_test.shape[0]/len(X)*100:.1f}%)")
        
        # Check target distribution across splits
        print(f"\n Target Distribution:")
        print(f"   Training mean: {y_train.mean():.4f} (std: {y_train.std():.4f})")
        print(f"   Validation mean: {y_val.mean():.4f} (std: {y_val.std():.4f})")
        print(f"   Test mean: {y_test.mean():.4f} (std: {y_test.std():.4f})")
        
        self.preprocessing_stats['data_split'] = {
            'train_size': len(X_train),
            'val_size': len(X_val),
            'test_size': len(X_test),
            'train_percentage': len(X_train)/len(X)*100,
            'val_percentage': len(X_val)/len(X)*100,
            'test_percentage': len(X_test)/len(X)*100,
            'target_stats': {
                'train_mean': y_train.mean(),
                'val_mean': y_val.mean(),
                'test_mean': y_test.mean()
            }
        }
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def save_preprocessing_pipeline(self, filepath: str = 'preprocessing_pipeline.pkl'):
        """
        Save the preprocessing pipeline for later use
        
        Args:
            filepath (str): Path to save the pipeline
        """
        pipeline = {
            'numerical_imputer': self.numerical_imputer,
            'categorical_imputer': self.categorical_imputer,
            'scaler': self.scaler,
            'encoders': self.encoders,
            'selected_features': self.selected_features,
            'numerical_features': self.numerical_features,
            'categorical_features': self.categorical_features,
            'target_column': self.target_column,
            'preprocessing_stats': self.preprocessing_stats
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(pipeline, f)
        
        print(f" Preprocessing pipeline saved to '{filepath}'")
    
    def load_preprocessing_pipeline(self, filepath: str = 'preprocessing_pipeline.pkl'):
        """
        Load a saved preprocessing pipeline
        
        Args:
            filepath (str): Path to the saved pipeline
        """
        with open(filepath, 'rb') as f:
            pipeline = pickle.load(f)
        
        self.numerical_imputer = pipeline['numerical_imputer']
        self.categorical_imputer = pipeline['categorical_imputer']
        self.scaler = pipeline['scaler']
        self.encoders = pipeline['encoders']
        self.selected_features = pipeline['selected_features']
        self.numerical_features = pipeline['numerical_features']
        self.categorical_features = pipeline['categorical_features']
        self.target_column = pipeline['target_column']
        self.preprocessing_stats = pipeline['preprocessing_stats']
        
        print(f" Preprocessing pipeline loaded from '{filepath}'")
    
    def transform_new_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Apply the fitted preprocessing pipeline to new data
        
        Args:
            data (pd.DataFrame): New data to transform
            
        Returns:
            pd.DataFrame: Transformed data
        """
        print("Applying preprocessing pipeline to new data...")
        
        if not all(comp is not None for comp in [self.numerical_imputer, self.categorical_imputer, self.scaler]):
            raise ValueError("Preprocessing pipeline not fitted. Run fit_transform first.")
        
        transformed_data = data.copy()
        
        # Apply same preprocessing steps
        existing_numerical = [f for f in self.numerical_features if f in transformed_data.columns]
        existing_categorical = [f for f in self.categorical_features if f in transformed_data.columns]
        
        # 1. Impute missing values
        if existing_numerical:
            transformed_data[existing_numerical] = self.numerical_imputer.transform(
                transformed_data[existing_numerical]
            )
        
        if existing_categorical:
            transformed_data[existing_categorical] = self.categorical_imputer.transform(
                transformed_data[existing_categorical]
            )
        
        # 2. Encode categorical features
        for feature in existing_categorical:
            if feature in self.encoders:
                le = self.encoders[feature]
                # Handle unknown categories
                transformed_data[f'{feature}_encoded'] = transformed_data[feature].map(
                    lambda x: le.transform([x])[0] if x in le.classes_ else 0
                )
        
        # 3. Scale numerical features
        if existing_numerical:
            scaled_features = [f'{feature}_scaled' for feature in existing_numerical]
            transformed_data[scaled_features] = self.scaler.transform(transformed_data[existing_numerical])
        
        print(f" Preprocessing applied to {len(transformed_data)} rows")
        
        return transformed_data
    
    def run_complete_preprocessing(self, data: pd.DataFrame, save_pipeline: bool = True) -> Tuple:
        """
        Run the complete preprocessing pipeline
        
        Args:
            data (pd.DataFrame): Raw input data
            save_pipeline (bool): Whether to save the pipeline
            
        Returns:
            Tuple: (X_train, X_val, X_test, y_train, y_val, y_test, processed_data)
        """
        print(" Starting Complete Data Preprocessing Pipeline")
        print("=" * 80)
        
        # Store original data
        self.original_data = data.copy()
        
        # Step 1: Analyze data quality
        self.analyze_data_quality(data)
        
        # Step 2: Clean data
        cleaned_data = self.clean_data(data)
        
        # Step 3: Handle missing values
        imputed_data = self.handle_missing_values(cleaned_data)
        
        # Step 4: Handle outliers
        outlier_handled_data = self.handle_outliers(imputed_data, method='iqr', action='cap')
        
        # Step 5: Encode categorical features
        encoded_data = self.encode_categorical_features(outlier_handled_data)
        
        # Step 6: Scale numerical features
        scaled_data = self.scale_features(encoded_data, method='minmax')
        
        # Step 7: Select features
        selected_features = self.select_features(scaled_data, method='correlation', top_k=15)
        
        # Step 8: Split data
        X_train, X_val, X_test, y_train, y_val, y_test = self.split_data(scaled_data)
        
        # Store processed data
        self.processed_data = scaled_data
        
        # Step 9: Save pipeline
        if save_pipeline:
            self.save_preprocessing_pipeline()
        
        print("\n" + "" * 20)
        print("PREPROCESSING COMPLETED SUCCESSFULLY!")
        print("" * 20)
        
        # Print summary
        print(f"\n Final Summary:")
        print(f"   Original data shape: {self.original_data.shape}")
        print(f"   Processed data shape: {self.processed_data.shape}")
        print(f"   Selected features: {len(self.selected_features)}")
        print(f"   Training samples: {len(X_train)}")
        print(f"   Validation samples: {len(X_val)}")
        print(f"   Test samples: {len(X_test)}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test, self.processed_data
    
    def get_preprocessing_report(self) -> str:
        """
        Generate a comprehensive preprocessing report
        
        Returns:
            str: Preprocessing report
        """
        report = "# Data Preprocessing Report\n\n"
        
        if 'data_quality' in self.preprocessing_stats:
            quality = self.preprocessing_stats['data_quality']
            report += f"## Data Quality Analysis\n"
            report += f"- **Original Shape**: {quality['total_rows']:,} rows × {quality['total_columns']} columns\n"
            
            # Missing values summary
            high_missing = sum(1 for mv in quality['missing_values'].values() if mv['percentage'] > 30)
            report += f"- **High Missing Features (>30%)**: {high_missing}\n"
            
            # Target issues
            if 'target_issues' in quality:
                target = quality['target_issues']
                report += f"- **Target Missing**: {target['missing_count']}\n"
                report += f"- **Target Outliers**: {target['extreme_values']}\n"
        
        if 'cleaning' in self.preprocessing_stats:
            cleaning = self.preprocessing_stats['cleaning']
            report += f"\n## Data Cleaning\n"
            report += f"- **Rows Removed**: {cleaning['rows_removed']} ({cleaning['removal_percentage']:.2f}%)\n"
            report += f"- **Final Dataset**: {cleaning['final_rows']:,} rows\n"
        
        if 'imputation' in self.preprocessing_stats:
            report += f"\n## Missing Value Imputation\n"
            report += f"- **Numerical Features**: Median imputation\n"
            report += f"- **Categorical Features**: Mode imputation\n"
        
        if 'outliers' in self.preprocessing_stats:
            outliers = self.preprocessing_stats['outliers']
            total_outliers = sum(o['outliers_count'] for o in outliers.values())
            report += f"\n## Outlier Handling\n"
            report += f"- **Total Outliers Handled**: {total_outliers}\n"
            report += f"- **Features Processed**: {len(outliers)}\n"
            report += f"- **Method**: IQR-based capping\n"
        
        if 'encoding' in self.preprocessing_stats:
            encoding = self.preprocessing_stats['encoding']
            report += f"\n## Categorical Encoding\n"
            report += f"- **Features Encoded**: {len(encoding)}\n"
            report += f"- **Method**: Label Encoding\n"
        
        if 'scaling' in self.preprocessing_stats:
            scaling = self.preprocessing_stats['scaling']
            report += f"\n## Feature Scaling\n"
            report += f"- **Features Scaled**: {len(scaling['features_scaled'])}\n"
            report += f"- **Method**: {scaling['method'].upper()}\n"
        
        if 'feature_selection' in self.preprocessing_stats:
            selection = self.preprocessing_stats['feature_selection']
            report += f"\n## Feature Selection\n"
            report += f"- **Available Features**: {selection['total_features']}\n"
            report += f"- **Selected Features**: {selection['selected_features']}\n"
            report += f"- **Selection Method**: {selection['method'].title()}\n"
        
        if 'data_split' in self.preprocessing_stats:
            split = self.preprocessing_stats['data_split']
            report += f"\n## Data Splitting\n"
            report += f"- **Training**: {split['train_size']} ({split['train_percentage']:.1f}%)\n"
            report += f"- **Validation**: {split['val_size']} ({split['val_percentage']:.1f}%)\n"
            report += f"- **Test**: {split['test_size']} ({split['test_percentage']:.1f}%)\n"
        
        return report

# Example usage
if __name__ == "__main__":
    from dataloader import DataLoader
    
    # Load data
    loader = DataLoader()
    main_data, _, _, _ = loader.load_all_data()
    
    if main_data is not None:
        # Run preprocessing
        preprocessor = DataPreprocessor()
        X_train, X_val, X_test, y_train, y_val, y_test, processed_data = preprocessor.run_complete_preprocessing(main_data)
        
        # Generate report
        report = preprocessor.get_preprocessing_report()
        with open('preprocessing_report.md', 'w') as f:
            f.write(report)
        
        print("\n Preprocessing report saved as 'preprocessing_report.md'")
        
        print(f"\n Ready for modeling!")
        print(f"   Training data: {X_train.shape}")
        print(f"   Features: {len(preprocessor.selected_features)}")
    else:
        print(" Failed to load data for preprocessing")