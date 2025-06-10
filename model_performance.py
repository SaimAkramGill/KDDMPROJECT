"""
Model Performance Module
Handles model training, evaluation, and performance analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
from sklearn.model_selection import cross_val_score, GridSearchCV, learning_curve
import pickle
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class ModelPerformanceAnalyzer:
    """
    Class for comprehensive model training and performance analysis
    """
    
    def __init__(self):
        """Initialize the model performance analyzer"""
        self.models = {}
        self.trained_models = {}
        self.performance_results = {}
        self.best_model = None
        self.best_model_name = None
        self.feature_importance = None
        self.learning_curves = {}
        
        # Initialize model configurations
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize different models with their configurations"""
        self.models = {
            'Random Forest': {
                'model': RandomForestRegressor(random_state=42),
                'params': {
                    'n_estimators': [50, 100, 150],
                    'max_depth': [5, 10, 15, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                }
            },
            'Gradient Boosting': {
                'model': GradientBoostingRegressor(random_state=42),
                'params': {
                    'n_estimators': [50, 100, 150],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'max_depth': [3, 5, 7],
                    'subsample': [0.8, 0.9, 1.0]
                }
            },
            'Linear Regression': {
                'model': LinearRegression(),
                'params': {}
            },
            'Ridge Regression': {
                'model': Ridge(random_state=42),
                'params': {
                    'alpha': [0.1, 1.0, 10.0, 100.0]
                }
            },
            'Lasso Regression': {
                'model': Lasso(random_state=42),
                'params': {
                    'alpha': [0.1, 1.0, 10.0, 100.0]
                }
            }
        }
    
    def train_models(self, X_train: pd.DataFrame, y_train: pd.Series, 
                    X_val: pd.DataFrame, y_val: pd.Series,
                    use_grid_search: bool = True, cv_folds: int = 5) -> Dict:
        """
        Train multiple models and compare their performance
        
        Args:
            X_train (pd.DataFrame): Training features
            y_train (pd.Series): Training target
            X_val (pd.DataFrame): Validation features
            y_val (pd.Series): Validation target
            use_grid_search (bool): Whether to use grid search for hyperparameter tuning
            cv_folds (int): Number of cross-validation folds
            
        Returns:
            Dict: Training results for all models
        """
        print("=" * 60)
        print("MODEL TRAINING AND COMPARISON")
        print("=" * 60)
        
        training_results = {}
        
        for model_name, model_config in self.models.items():
            print(f"\n🔄 Training {model_name}...")
            print("-" * 40)
            
            try:
                model = model_config['model']
                params = model_config['params']
                
                # Hyperparameter tuning if parameters are specified
                if use_grid_search and params:
                    print(f"   Performing grid search with {cv_folds}-fold CV...")
                    
                    grid_search = GridSearchCV(
                        model, params, cv=cv_folds, 
                        scoring='neg_mean_absolute_error',
                        n_jobs=-1, verbose=0
                    )
                    
                    grid_search.fit(X_train, y_train)
                    best_model = grid_search.best_estimator_
                    best_params = grid_search.best_params_
                    
                    print(f"   Best parameters: {best_params}")
                else:
                    best_model = model
                    best_params = {}
                    print(f"   Using default parameters")
                
                # Train the model
                best_model.fit(X_train, y_train)
                
                # Make predictions
                train_pred = best_model.predict(X_train)
                val_pred = best_model.predict(X_val)
                
                # Calculate metrics
                train_metrics = self._calculate_metrics(y_train, train_pred)
                val_metrics = self._calculate_metrics(y_val, val_pred)
                
                # Cross-validation scores
                cv_scores = cross_val_score(
                    best_model, X_train, y_train, 
                    cv=cv_folds, scoring='neg_mean_absolute_error'
                )
                cv_mae = -cv_scores.mean()
                cv_std = cv_scores.std()
                
                # Store results
                training_results[model_name] = {
                    'model': best_model,
                    'best_params': best_params,
                    'train_metrics': train_metrics,
                    'val_metrics': val_metrics,
                    'cv_mae': cv_mae,
                    'cv_std': cv_std,
                    'cv_scores': cv_scores
                }
                
                # Print results
                print(f"   Training MAE: {train_metrics['mae']:.4f}")
                print(f"   Validation MAE: {val_metrics['mae']:.4f}")
                print(f"   CV MAE: {cv_mae:.4f} ± {cv_std:.4f}")
                print(f"   Validation R²: {val_metrics['r2']:.4f}")
                
                # Store trained model
                self.trained_models[model_name] = best_model
                
            except Exception as e:
                print(f"   ❌ Error training {model_name}: {str(e)}")
                continue
        
        # Select best model based on validation MAE
        if training_results:
            best_model_name = min(training_results.keys(), 
                                key=lambda k: training_results[k]['val_metrics']['mae'])
            
            self.best_model_name = best_model_name
            self.best_model = training_results[best_model_name]['model']
            
            print(f"\n Best Model: {best_model_name}")
            print(f"   Validation MAE: {training_results[best_model_name]['val_metrics']['mae']:.4f}")
            print(f"   Validation R²: {training_results[best_model_name]['val_metrics']['r2']:.4f}")
        
        self.performance_results = training_results
        return training_results
    
    def _calculate_metrics(self, y_true: pd.Series, y_pred: np.ndarray) -> Dict:
        """
        Calculate comprehensive performance metrics
        
        Args:
            y_true (pd.Series): True values
            y_pred (np.ndarray): Predicted values
            
        Returns:
            Dict: Performance metrics
        """
        return {
            'mae': mean_absolute_error(y_true, y_pred),
            'mse': mean_squared_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'r2': r2_score(y_true, y_pred),
            'mape': mean_absolute_percentage_error(y_true, y_pred) * 100  # Convert to percentage
        }
    
    def evaluate_on_test_set(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict:
        """
        Evaluate all trained models on test set
        
        Args:
            X_test (pd.DataFrame): Test features
            y_test (pd.Series): Test target
            
        Returns:
            Dict: Test set evaluation results
        """
        print("\n" + "=" * 60)
        print("TEST SET EVALUATION")
        print("=" * 60)
        
        test_results = {}
        
        for model_name, model in self.trained_models.items():
            print(f"\n Evaluating {model_name} on test set...")
            
            # Make predictions
            test_pred = model.predict(X_test)
            
            # Calculate metrics
            test_metrics = self._calculate_metrics(y_test, test_pred)
            
            test_results[model_name] = {
                'metrics': test_metrics,
                'predictions': test_pred
            }
            
            print(f"   Test MAE: {test_metrics['mae']:.4f}")
            print(f"   Test RMSE: {test_metrics['rmse']:.4f}")
            print(f"   Test R²: {test_metrics['r2']:.4f}")
            print(f"   Test MAPE: {test_metrics['mape']:.2f}%")
        
        return test_results
    
    def analyze_feature_importance(self, feature_names: List[str]) -> pd.DataFrame:
        """
        Analyze feature importance for tree-based models
        
        Args:
            feature_names (List[str]): List of feature names
            
        Returns:
            pd.DataFrame: Feature importance analysis
        """
        print("\n" + "=" * 60)
        print("FEATURE IMPORTANCE ANALYSIS")
        print("=" * 60)
        
        importance_results = {}
        
        for model_name, model in self.trained_models.items():
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                
                # Create importance dataframe
                importance_df = pd.DataFrame({
                    'Feature': feature_names,
                    'Importance': importances
                }).sort_values('Importance', ascending=False)
                
                importance_results[model_name] = importance_df
                
                print(f"\n🔍 {model_name} - Top 10 Features:")
                for i, (_, row) in enumerate(importance_df.head(10).iterrows()):
                    print(f"   {i+1:2d}. {row['Feature']}: {row['Importance']:.4f}")
        
        # Store feature importance for best model
        if self.best_model_name in importance_results:
            self.feature_importance = importance_results[self.best_model_name]
            print(f"\n Feature importance stored for best model: {self.best_model_name}")
        
        return importance_results
    
    def generate_learning_curves(self, X_train: pd.DataFrame, y_train: pd.Series, 
                                cv_folds: int = 5) -> Dict:
        """
        Generate learning curves for model analysis
        
        Args:
            X_train (pd.DataFrame): Training features
            y_train (pd.Series): Training target
            cv_folds (int): Number of cross-validation folds
            
        Returns:
            Dict: Learning curve results
        """
        print("\n" + "=" * 60)
        print("GENERATING LEARNING CURVES")
        print("=" * 60)
        
        learning_curve_results = {}
        
        for model_name, model in self.trained_models.items():
            print(f"\n Generating learning curve for {model_name}...")
            
            try:
                # Generate learning curve
                train_sizes, train_scores, val_scores = learning_curve(
                    model, X_train, y_train, cv=cv_folds, 
                    scoring='neg_mean_absolute_error',
                    train_sizes=np.linspace(0.1, 1.0, 10),
                    n_jobs=-1
                )
                
                # Convert to positive MAE scores
                train_scores = -train_scores
                val_scores = -val_scores
                
                # Calculate means and standard deviations
                train_mean = np.mean(train_scores, axis=1)
                train_std = np.std(train_scores, axis=1)
                val_mean = np.mean(val_scores, axis=1)
                val_std = np.std(val_scores, axis=1)
                
                learning_curve_results[model_name] = {
                    'train_sizes': train_sizes,
                    'train_mean': train_mean,
                    'train_std': train_std,
                    'val_mean': val_mean,
                    'val_std': val_std
                }
                
                print(f"    Learning curve generated")
                
            except Exception as e:
                print(f"    Error generating learning curve: {str(e)}")
        
        self.learning_curves = learning_curve_results
        return learning_curve_results
    
    def create_performance_visualizations(self, X_test: pd.DataFrame, y_test: pd.Series,
                                        save_path: str = 'model_performance.png'):
        """
        Create comprehensive performance visualizations
        
        Args:
            X_test (pd.DataFrame): Test features
            y_test (pd.Series): Test target
            save_path (str): Path to save visualizations
        """
        print(f"\n Creating performance visualizations...")
        
        # Set up the figure
        fig = plt.figure(figsize=(20, 16))
        
        # 1. Model comparison (validation MAE)
        plt.subplot(3, 3, 1)
        if self.performance_results:
            model_names = list(self.performance_results.keys())
            val_maes = [self.performance_results[name]['val_metrics']['mae'] for name in model_names]
            
            bars = plt.bar(range(len(model_names)), val_maes, 
                          color=['gold' if name == self.best_model_name else 'lightblue' for name in model_names])
            plt.title('Model Comparison (Validation MAE)', fontweight='bold')
            plt.xlabel('Models')
            plt.ylabel('Mean Absolute Error')
            plt.xticks(range(len(model_names)), model_names, rotation=45, ha='right')
            
            # Add value labels
            for i, bar in enumerate(bars):
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                        f'{height:.4f}', ha='center', va='bottom', fontsize=9)
        
        # 2. R² Score comparison
        plt.subplot(3, 3, 2)
        if self.performance_results:
            r2_scores = [self.performance_results[name]['val_metrics']['r2'] for name in model_names]
            
            bars = plt.bar(range(len(model_names)), r2_scores,
                          color=['gold' if name == self.best_model_name else 'lightcoral' for name in model_names])
            plt.title('Model Comparison (Validation R²)', fontweight='bold')
            plt.xlabel('Models')
            plt.ylabel('R² Score')
            plt.xticks(range(len(model_names)), model_names, rotation=45, ha='right')
            
            # Add value labels
            for i, bar in enumerate(bars):
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{height:.3f}', ha='center', va='bottom', fontsize=9)
        
        # 3. Actual vs Predicted (Best Model)
        plt.subplot(3, 3, 3)
        if self.best_model is not None:
            test_pred = self.best_model.predict(X_test)
            
            plt.scatter(y_test, test_pred, alpha=0.6, s=20)
            
            # Perfect prediction line
            min_val = min(y_test.min(), test_pred.min())
            max_val = max(y_test.max(), test_pred.max())
            plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
            
            plt.xlabel('Actual Values')
            plt.ylabel('Predicted Values')
            plt.title(f'Actual vs Predicted ({self.best_model_name})', fontweight='bold')
            plt.legend()
            
            # Add R² to plot
            r2 = r2_score(y_test, test_pred)
            plt.text(0.05, 0.95, f'R² = {r2:.4f}', transform=plt.gca().transAxes,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # 4. Residuals plot
        plt.subplot(3, 3, 4)
        if self.best_model is not None:
            test_pred = self.best_model.predict(X_test)
            residuals = y_test - test_pred
            
            plt.scatter(test_pred, residuals, alpha=0.6, s=20)
            plt.axhline(y=0, color='r', linestyle='--')
            plt.xlabel('Predicted Values')
            plt.ylabel('Residuals')
            plt.title(f'Residuals Plot ({self.best_model_name})', fontweight='bold')
        
        # 5. Feature importance (if available)
        plt.subplot(3, 3, 5)
        if self.feature_importance is not None:
            top_features = self.feature_importance.head(10)
            
            plt.barh(range(len(top_features)), top_features['Importance'].values)
            plt.yticks(range(len(top_features)), top_features['Feature'].values)
            plt.xlabel('Importance')
            plt.title(f'Top 10 Feature Importance ({self.best_model_name})', fontweight='bold')
            plt.gca().invert_yaxis()
        
        # 6. Cross-validation scores
        plt.subplot(3, 3, 6)
        if self.performance_results:
            cv_means = [self.performance_results[name]['cv_mae'] for name in model_names]
            cv_stds = [self.performance_results[name]['cv_std'] for name in model_names]
            
            bars = plt.bar(range(len(model_names)), cv_means, yerr=cv_stds, capsize=5,
                          color=['gold' if name == self.best_model_name else 'lightgreen' for name in model_names])
            plt.title('Cross-Validation MAE (±1 std)', fontweight='bold')
            plt.xlabel('Models')
            plt.ylabel('CV MAE')
            plt.xticks(range(len(model_names)), model_names, rotation=45, ha='right')
        
        # 7. Learning curve (best model)
        plt.subplot(3, 3, 7)
        if self.best_model_name in self.learning_curves:
            lc_data = self.learning_curves[self.best_model_name]
            
            plt.plot(lc_data['train_sizes'], lc_data['train_mean'], 'o-', label='Training', color='blue')
            plt.fill_between(lc_data['train_sizes'], 
                           lc_data['train_mean'] - lc_data['train_std'],
                           lc_data['train_mean'] + lc_data['train_std'], alpha=0.3, color='blue')
            
            plt.plot(lc_data['train_sizes'], lc_data['val_mean'], 'o-', label='Validation', color='orange')
            plt.fill_between(lc_data['train_sizes'], 
                           lc_data['val_mean'] - lc_data['val_std'],
                           lc_data['val_mean'] + lc_data['val_std'], alpha=0.3, color='orange')
            
            plt.xlabel('Training Set Size')
            plt.ylabel('MAE')
            plt.title(f'Learning Curve ({self.best_model_name})', fontweight='bold')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        # 8. Error distribution
        plt.subplot(3, 3, 8)
        if self.best_model is not None:
            test_pred = self.best_model.predict(X_test)
            errors = y_test - test_pred
            
            plt.hist(errors, bins=30, alpha=0.7, edgecolor='black')
            plt.axvline(x=0, color='r', linestyle='--', label='Zero Error')
            plt.xlabel('Prediction Error')
            plt.ylabel('Frequency')
            plt.title(f'Error Distribution ({self.best_model_name})', fontweight='bold')
            plt.legend()
        
        # 9. Model complexity comparison
        plt.subplot(3, 3, 9)
        if self.performance_results:
            # Create a simple complexity score based on model type
            complexity_scores = []
            for name in model_names:
                if 'Linear' in name:
                    complexity_scores.append(1)
                elif 'Ridge' in name or 'Lasso' in name:
                    complexity_scores.append(2)
                elif 'Random Forest' in name:
                    complexity_scores.append(4)
                elif 'Gradient' in name:
                    complexity_scores.append(5)
                else:
                    complexity_scores.append(3)
            
            val_maes = [self.performance_results[name]['val_metrics']['mae'] for name in model_names]
            
            colors = ['gold' if name == self.best_model_name else 'lightblue' for name in model_names]
            scatter = plt.scatter(complexity_scores, val_maes, c=colors, s=100, alpha=0.7)
            
            # Add model name labels
            for i, name in enumerate(model_names):
                plt.annotate(name, (complexity_scores[i], val_maes[i]), 
                           xytext=(5, 5), textcoords='offset points', fontsize=8)
            
            plt.xlabel('Model Complexity')
            plt.ylabel('Validation MAE')
            plt.title('Model Complexity vs Performance', fontweight='bold')
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f" Performance visualizations saved as '{save_path}'")
    
    def save_best_model(self, filepath: str = 'best_model.pkl'):
        """
        Save the best model to file
        
        Args:
            filepath (str): Path to save the model
        """
        if self.best_model is not None:
            model_package = {
                'model': self.best_model,
                'model_name': self.best_model_name,
                'feature_importance': self.feature_importance,
                'performance_metrics': self.performance_results[self.best_model_name] if self.best_model_name in self.performance_results else None
            }
            
            with open(filepath, 'wb') as f:
                pickle.dump(model_package, f)
            
            print(f" Best model ({self.best_model_name}) saved to '{filepath}'")
        else:
            print(" No trained model available to save")
    
    def load_model(self, filepath: str = 'best_model.pkl'):
        """
        Load a saved model
        
        Args:
            filepath (str): Path to the saved model
        """
        with open(filepath, 'rb') as f:
            model_package = pickle.load(f)
        
        self.best_model = model_package['model']
        self.best_model_name = model_package['model_name']
        self.feature_importance = model_package.get('feature_importance')
        
        print(f" Model ({self.best_model_name}) loaded from '{filepath}'")
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions using the best model
        
        Args:
            X (pd.DataFrame): Features for prediction
            
        Returns:
            np.ndarray: Predictions
        """
        if self.best_model is None:
            raise ValueError("No trained model available. Train a model first.")
        
        return self.best_model.predict(X)
    
    def generate_performance_report(self) -> str:
        """
        Generate comprehensive performance report
        
        Returns:
            str: Performance report
        """
        report = "# Model Performance Report\n\n"
        
        if self.performance_results:
            report += "## Model Comparison\n\n"
            
            # Create comparison table
            comparison_data = []
            for model_name, results in self.performance_results.items():
                comparison_data.append({
                    'Model': model_name,
                    'Validation MAE': f"{results['val_metrics']['mae']:.4f}",
                    'Validation R²': f"{results['val_metrics']['r2']:.4f}",
                    'CV MAE': f"{results['cv_mae']:.4f} ± {results['cv_std']:.4f}",
                    'Training MAE': f"{results['train_metrics']['mae']:.4f}"
                })
            
            comparison_df = pd.DataFrame(comparison_data)
            report += comparison_df.to_markdown(index=False) + "\n\n"
            
            # Best model details
            if self.best_model_name:
                best_results = self.performance_results[self.best_model_name]
                report += f"## Best Model: {self.best_model_name}\n\n"
                report += f"### Performance Metrics\n"
                report += f"- **Validation MAE**: {best_results['val_metrics']['mae']:.4f}\n"
                report += f"- **Validation RMSE**: {best_results['val_metrics']['rmse']:.4f}\n"
                report += f"- **Validation R²**: {best_results['val_metrics']['r2']:.4f}\n"
                report += f"- **Validation MAPE**: {best_results['val_metrics']['mape']:.2f}%\n"
                report += f"- **Cross-Validation MAE**: {best_results['cv_mae']:.4f} ± {best_results['cv_std']:.4f}\n\n"
                
                if best_results['best_params']:
                    report += f"### Best Hyperparameters\n"
                    for param, value in best_results['best_params'].items():
                        report += f"- **{param}**: {value}\n"
                    report += "\n"
        
        # Feature importance
        if self.feature_importance is not None:
            report += "## Top 10 Feature Importance\n\n"
            for i, (_, row) in enumerate(self.feature_importance.head(10).iterrows()):
                report += f"{i+1}. **{row['Feature']}**: {row['Importance']:.4f}\n"
            report += "\n"
        
        # Model insights
        report += "## Model Insights\n\n"
        if self.best_model_name:
            if 'Random Forest' in self.best_model_name:
                report += "- **Random Forest** provides good performance with feature importance insights\n"
                report += "- Robust to outliers and handles non-linear relationships well\n"
                report += "- Feature importance helps identify key predictors\n"
            elif 'Gradient' in self.best_model_name:
                report += "- **Gradient Boosting** excels at capturing complex patterns\n"
                report += "- Sequential learning approach improves predictions iteratively\n"
                report += "- May require careful tuning to avoid overfitting\n"
            elif 'Linear' in self.best_model_name:
                report += "- **Linear Regression** provides interpretable baseline performance\n"
                report += "- Assumes linear relationships between features and target\n"
                report += "- Fast training and prediction times\n"
        
        return report
    
    def run_complete_analysis(self, X_train: pd.DataFrame, y_train: pd.Series,
                            X_val: pd.DataFrame, y_val: pd.Series,
                            X_test: pd.DataFrame, y_test: pd.Series,
                            feature_names: List[str], save_results: bool = True) -> Dict:
        """
        Run complete model performance analysis
        
        Args:
            X_train, y_train: Training data
            X_val, y_val: Validation data
            X_test, y_test: Test data
            feature_names: List of feature names
            save_results: Whether to save results
            
        Returns:
            Dict: Complete analysis results
        """
        print(" Starting Complete Model Performance Analysis")
        print("=" * 80)
        
        # Step 1: Train models
        training_results = self.train_models(X_train, y_train, X_val, y_val)
        
        # Step 2: Evaluate on test set
        test_results = self.evaluate_on_test_set(X_test, y_test)
        
        # Step 3: Analyze feature importance
        importance_results = self.analyze_feature_importance(feature_names)
        
        # Step 4: Generate learning curves
        learning_curves = self.generate_learning_curves(X_train, y_train)
        
        # Step 5: Create visualizations
        self.create_performance_visualizations(X_test, y_test)
        
        # Step 6: Save results
        if save_results:
            self.save_best_model()
            
            # Save performance report
            report = self.generate_performance_report()
            with open('model_performance_report.md', 'w') as f:
                f.write(report)
            print("📄 Performance report saved as 'model_performance_report.md'")
        
        print("\n" + "" * 20)
        print("MODEL ANALYSIS COMPLETED SUCCESSFULLY!")
        print("" * 20)
        
        analysis_results = {
            'training_results': training_results,
            'test_results': test_results,
            'feature_importance': importance_results,
            'learning_curves': learning_curves,
            'best_model': self.best_model,
            'best_model_name': self.best_model_name
        }
        
        print(f"\n Analysis Summary:")
        print(f"   Models trained: {len(training_results)}")
        print(f"   Best model: {self.best_model_name}")
        if self.best_model_name in test_results:
            best_test_mae = test_results[self.best_model_name]['metrics']['mae']
            best_test_r2 = test_results[self.best_model_name]['metrics']['r2']
            print(f"   Test MAE: {best_test_mae:.4f}")
            print(f"   Test R²: {best_test_r2:.4f}")
        
        return analysis_results

# Example usage
if __name__ == "__main__":
    from dataloader import DataLoader
    from preprocessing import DataPreprocessor
    
    # Load and preprocess data
    loader = DataLoader()
    main_data, _, _, _ = loader.load_all_data()
    
    if main_data is not None:
        # Preprocess data
        preprocessor = DataPreprocessor()
        X_train, X_val, X_test, y_train, y_val, y_test, _ = preprocessor.run_complete_preprocessing(main_data)
        
        # Run model analysis
        analyzer = ModelPerformanceAnalyzer()
        results = analyzer.run_complete_analysis(
            X_train, y_train, X_val, y_val, X_test, y_test,
            preprocessor.selected_features
        )
        
        print(f"\n Model Performance Analysis Complete!")
        print(f"   Best model ready for predictions: {analyzer.best_model_name}")
    else:
        print(" Failed to load data for model analysis")