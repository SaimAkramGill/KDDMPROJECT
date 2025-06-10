"""
Task 3 Module: Unbeatable Villain Analysis
Analyzes why the Task 3 villain is unbeatable and what features make them invincible
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

class Task3VillainAnalyzer:
    """
    Class for analyzing the unbeatable villain in Task 3
    """
    
    def __init__(self, model, preprocessor, main_dataset: pd.DataFrame):
        """
        Initialize Task 3 analyzer
        
        Args:
            model: Trained model for predictions
            preprocessor: Fitted data preprocessor
            main_dataset (pd.DataFrame): Original training dataset for comparison
        """
        self.model = model
        self.preprocessor = preprocessor
        self.main_dataset = main_dataset
        self.task3_villain = None
        self.villain_analysis = None
        self.feature_comparison = None
        
    def load_task3_data(self, villain_data: pd.DataFrame):
        """
        Load Task 3 villain data
        
        Args:
            villain_data (pd.DataFrame): Task 3 villain data
        """
        self.task3_villain = villain_data.copy()
        
        print("=" * 60)
        print("TASK 3: UNBEATABLE VILLAIN ANALYSIS")
        print("=" * 60)
        
        if len(self.task3_villain) > 0:
            villain = self.task3_villain.iloc[0]
            villain_name = villain.get('name', 'Unknown')
            villain_role = villain.get('role', 'Unknown')
            villain_win_prob = villain.get('win_prob', 'Unknown')
            
            print(f"🦹 Analyzing unbeatable villain:")
            print(f"   Name: {villain_name}")
            print(f"   Role: {villain_role}")
            print(f"   Win Probability: {villain_win_prob}")
            
            if villain_win_prob == 1.0 or str(villain_win_prob) == '1.0':
                print(f"   ✅ Confirmed: This villain is truly unbeatable!")
            else:
                print(f"   ⚠️  Note: Win probability is {villain_win_prob}, not exactly 1.0")
        else:
            print("❌ No villain data found")
    
    def calculate_dataset_statistics(self) -> Dict:
        """
        Calculate statistics from the main dataset for comparison
        
        Returns:
            Dict: Dataset statistics
        """
        print(f"\n📊 Calculating dataset statistics for comparison...")
        
        stats = {
            'numerical_stats': {},
            'categorical_stats': {},
            'win_prob_stats': {}
        }
        
        # Numerical feature statistics
        numerical_features = self.preprocessor.numerical_features
        for feature in numerical_features:
            if feature in self.main_dataset.columns:
                feature_data = self.main_dataset[feature].dropna()
                if len(feature_data) > 0:
                    stats['numerical_stats'][feature] = {
                        'mean': feature_data.mean(),
                        'median': feature_data.median(),
                        'std': feature_data.std(),
                        'min': feature_data.min(),
                        'max': feature_data.max(),
                        'q1': feature_data.quantile(0.25),
                        'q3': feature_data.quantile(0.75)
                    }
        
        # Categorical feature statistics
        categorical_features = self.preprocessor.categorical_features
        for feature in categorical_features:
            if feature in self.main_dataset.columns:
                value_counts = self.main_dataset[feature].value_counts()
                stats['categorical_stats'][feature] = {
                    'mode': value_counts.index[0] if len(value_counts) > 0 else None,
                    'unique_count': self.main_dataset[feature].nunique(),
                    'value_counts': value_counts.to_dict()
                }
        
        # Win probability statistics
        win_prob_data = self.main_dataset['win_prob'].dropna()
        stats['win_prob_stats'] = {
            'mean': win_prob_data.mean(),
            'median': win_prob_data.median(),
            'std': win_prob_data.std(),
            'min': win_prob_data.min(),
            'max': win_prob_data.max(),
            'high_performers_count': (win_prob_data > 0.9).sum(),
            'perfect_performers_count': (win_prob_data >= 1.0).sum()
        }
        
        print(f"✅ Dataset statistics calculated")
        print(f"   Numerical features: {len(stats['numerical_stats'])}")
        print(f"   Categorical features: {len(stats['categorical_stats'])}")
        print(f"   Characters with >90% win rate: {stats['win_prob_stats']['high_performers_count']}")
        print(f"   Perfect performers (≥100%): {stats['win_prob_stats']['perfect_performers_count']}")
        
        return stats
    
    def analyze_villain_features(self) -> Dict:
        """
        Analyze the villain's features compared to the dataset
        
        Returns:
            Dict: Feature analysis results
        """
        print(f"\n🔍 Analyzing villain's features...")
        
        if self.task3_villain is None or len(self.task3_villain) == 0:
            raise ValueError("Task 3 villain data not loaded")
        
        villain = self.task3_villain.iloc[0]
        dataset_stats = self.calculate_dataset_statistics()
        
        analysis = {
            'numerical_comparison': {},
            'categorical_analysis': {},
            'extreme_features': [],
            'strategic_advantages': []
        }
        
        print(f"\n📋 NUMERICAL FEATURES COMPARISON")
        print("-" * 50)
        
        # Analyze numerical features
        for feature in self.preprocessor.numerical_features:
            if feature in self.task3_villain.columns and feature in dataset_stats['numerical_stats']:
                villain_value = villain[feature]
                stats = dataset_stats['numerical_stats'][feature]
                
                if pd.notnull(villain_value):
                    # Calculate percentile
                    feature_data = self.main_dataset[feature].dropna()
                    percentile = (feature_data <= villain_value).mean() * 100
                    
                    # Calculate deviation from mean
                    mean_value = stats['mean']
                    percent_diff = ((villain_value - mean_value) / mean_value) * 100 if mean_value != 0 else 0
                    
                    # Determine if this is an extreme value
                    is_extreme = (villain_value < stats['q1'] - 1.5 * (stats['q3'] - stats['q1'])) or \
                                (villain_value > stats['q3'] + 1.5 * (stats['q3'] - stats['q1']))
                    
                    comparison = {
                        'villain_value': villain_value,
                        'dataset_mean': mean_value,
                        'dataset_median': stats['median'],
                        'percent_diff_from_mean': percent_diff,
                        'percentile': percentile,
                        'is_extreme': is_extreme,
                        'z_score': (villain_value - mean_value) / stats['std'] if stats['std'] > 0 else 0
                    }
                    
                    analysis['numerical_comparison'][feature] = comparison
                    
                    # Print comparison
                    direction = "higher" if percent_diff > 0 else "lower"
                    extreme_marker = " 🔥" if is_extreme else ""
                    percentile_desc = f"({percentile:.1f}th percentile)"
                    
                    print(f"   {feature}:")
                    print(f"     Villain: {villain_value}")
                    print(f"     Dataset avg: {mean_value:.2f}")
                    print(f"     Difference: {percent_diff:+.2f}% {direction} {percentile_desc}{extreme_marker}")
                    
                    # Track extreme features
                    if abs(percent_diff) > 20 or is_extreme:
                        analysis['extreme_features'].append({
                            'feature': feature,
                            'type': 'numerical',
                            'deviation': percent_diff,
                            'significance': 'extreme' if is_extreme else 'high'
                        })
        
        print(f"\n📋 CATEGORICAL FEATURES ANALYSIS")
        print("-" * 50)
        
        # Analyze categorical features
        for feature in self.preprocessor.categorical_features:
            if feature in self.task3_villain.columns and feature in dataset_stats['categorical_stats']:
                villain_value = villain[feature]
                stats = dataset_stats['categorical_stats'][feature]
                
                if pd.notnull(villain_value):
                    # Calculate frequency of this value in dataset
                    value_counts = stats['value_counts']
                    villain_value_count = value_counts.get(villain_value, 0)
                    total_count = sum(value_counts.values())
                    frequency_pct = (villain_value_count / total_count) * 100 if total_count > 0 else 0
                    
                    # Determine rarity
                    if frequency_pct == 0:
                        rarity = "Unique/Unknown"
                    elif frequency_pct < 5:
                        rarity = "Very Rare"
                    elif frequency_pct < 15:
                        rarity = "Rare"
                    elif frequency_pct < 30:
                        rarity = "Uncommon"
                    else:
                        rarity = "Common"
                    
                    categorical_info = {
                        'villain_value': villain_value,
                        'frequency_count': villain_value_count,
                        'frequency_percentage': frequency_pct,
                        'rarity': rarity,
                        'most_common': stats['mode'],
                        'is_mode': villain_value == stats['mode']
                    }
                    
                    analysis['categorical_analysis'][feature] = categorical_info
                    
                    # Print analysis
                    rarity_marker = " ⭐" if rarity in ["Unique/Unknown", "Very Rare"] else ""
                    mode_marker = " 👑" if categorical_info['is_mode'] else ""
                    
                    print(f"   {feature}:")
                    print(f"     Villain: {villain_value}")
                    print(f"     Frequency: {villain_value_count}/{total_count} ({frequency_pct:.1f}%)")
                    print(f"     Rarity: {rarity}{rarity_marker}{mode_marker}")
                    
                    # Track strategic advantages
                    if rarity in ["Unique/Unknown", "Very Rare"] or categorical_info['is_mode']:
                        analysis['strategic_advantages'].append({
                            'feature': feature,
                            'type': 'categorical',
                            'value': villain_value,
                            'advantage_type': 'unique' if rarity in ["Unique/Unknown", "Very Rare"] else 'popular'
                        })
        
        self.feature_comparison = analysis
        return analysis
    
    def predict_villain_probability(self) -> Dict:
        """
        Use the model to predict the villain's win probability
        
        Returns:
            Dict: Prediction results
        """
        print(f"\n🎯 Predicting villain's win probability using trained model...")
        
        # Preprocess villain data
        processed_villain = self.preprocessor.transform_new_data(self.task3_villain)
        
        # Extract features for prediction
        X_villain = processed_villain[self.preprocessor.selected_features]
        
        # Make prediction
        predicted_prob = self.model.predict(X_villain)[0]
        actual_prob = self.task3_villain.iloc[0].get('win_prob', 1.0)
        
        prediction_analysis = {
            'predicted_probability': predicted_prob,
            'actual_probability': actual_prob,
            'prediction_error': abs(predicted_prob - actual_prob),
            'model_captures_unbeatable': predicted_prob >= 0.95
        }
        
        print(f"   Model prediction: {predicted_prob:.4f}")
        print(f"   Actual probability: {actual_prob}")
        print(f"   Prediction error: {prediction_analysis['prediction_error']:.4f}")
        
        if prediction_analysis['model_captures_unbeatable']:
            print(f"   ✅ Model successfully identifies this as a top-tier character")
        else:
            print(f"   ⚠️  Model underestimates this character's true strength")
        
        return prediction_analysis
    
    def analyze_feature_importance_impact(self) -> Dict:
        """
        Analyze how the villain's features align with model's feature importance
        
        Returns:
            Dict: Feature importance impact analysis
        """
        print(f"\n🎯 Analyzing feature importance impact...")
        
        if not hasattr(self.model, 'feature_importances_'):
            print("   ⚠️  Model doesn't provide feature importance")
            return {}
        
        # Get feature importance
        feature_importance = dict(zip(self.preprocessor.selected_features, self.model.feature_importances_))
        
        # Analyze villain's performance on important features
        importance_analysis = {
            'top_features_analysis': {},
            'villain_advantage_score': 0,
            'key_strengths': [],
            'explanation_factors': []
        }
        
        # Sort features by importance
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        
        print(f"   Analyzing top 10 most important features:")
        print("-" * 40)
        
        # Process villain data to get feature values
        processed_villain = self.preprocessor.transform_new_data(self.task3_villain)
        
        for i, (feature, importance) in enumerate(sorted_features[:10]):
            if feature in processed_villain.columns:
                villain_feature_value = processed_villain[feature].iloc[0]
                
                # Get the original feature name for better interpretation
                original_feature = feature.replace('_scaled', '').replace('_encoded', '')
                
                # For numerical features, compare to dataset mean
                if feature.endswith('_scaled'):
                    # Scaled features should be between 0 and 1
                    # Values closer to 1 are generally better (higher than average)
                    performance_score = villain_feature_value
                    interpretation = "High" if villain_feature_value > 0.7 else "Medium" if villain_feature_value > 0.3 else "Low"
                    
                elif feature.endswith('_encoded'):
                    # For encoded categorical features, we need to interpret differently
                    # We'll use the raw categorical value for interpretation
                    if original_feature in self.task3_villain.columns:
                        raw_value = self.task3_villain[original_feature].iloc[0]
                        performance_score = 0.5  # Neutral for categorical
                        interpretation = f"Value: {raw_value}"
                    else:
                        performance_score = 0.5
                        interpretation = f"Encoded: {villain_feature_value}"
                else:
                    performance_score = 0.5
                    interpretation = "Unknown"
                
                feature_analysis = {
                    'feature_name': original_feature,
                    'importance': importance,
                    'villain_value': villain_feature_value,
                    'performance_score': performance_score,
                    'interpretation': interpretation
                }
                
                importance_analysis['top_features_analysis'][feature] = feature_analysis
                
                # Add to advantage score (weighted by importance)
                importance_analysis['villain_advantage_score'] += importance * performance_score
                
                print(f"   {i+1:2d}. {original_feature}")
                print(f"       Importance: {importance:.4f}")
                print(f"       Villain value: {villain_feature_value:.4f}")
                print(f"       Performance: {interpretation}")
                
                # Identify key strengths
                if importance > 0.05 and performance_score > 0.6:  # High importance, high performance
                    importance_analysis['key_strengths'].append({
                        'feature': original_feature,
                        'importance': importance,
                        'performance': performance_score
                    })
        
        print(f"\n   🏆 Villain's advantage score: {importance_analysis['villain_advantage_score']:.4f}")
        print(f"   🎯 Key strengths identified: {len(importance_analysis['key_strengths'])}")
        
        return importance_analysis
    
    def explain_unbeatable_status(self) -> Dict:
        """
        Generate comprehensive explanation for why the villain is unbeatable
        
        Returns:
            Dict: Explanation analysis
        """
        print(f"\n💡 Generating explanation for unbeatable status...")
        
        if self.feature_comparison is None:
            self.analyze_villain_features()
        
        explanation = {
            'primary_factors': [],
            'secondary_factors': [],
            'strategic_elements': [],
            'conclusion': ""
        }
        
        # Primary factors (extreme numerical advantages)
        for feature_info in self.feature_comparison['extreme_features']:
            if feature_info['type'] == 'numerical' and abs(feature_info['deviation']) > 30:
                explanation['primary_factors'].append({
                    'factor': feature_info['feature'],
                    'type': 'numerical_extreme',
                    'deviation': feature_info['deviation'],
                    'description': f"Exceptional {feature_info['feature']}"
                })
        
        # Secondary factors (significant advantages)
        for feature, comparison in self.feature_comparison['numerical_comparison'].items():
            if 10 < abs(comparison['percent_diff_from_mean']) <= 30:
                explanation['secondary_factors'].append({
                    'factor': feature,
                    'type': 'numerical_advantage',
                    'deviation': comparison['percent_diff_from_mean'],
                    'percentile': comparison['percentile']
                })
        
        # Strategic elements (categorical advantages)
        for advantage in self.feature_comparison['strategic_advantages']:
            explanation['strategic_elements'].append({
                'factor': advantage['feature'],
                'type': advantage['type'],
                'value': advantage['value'],
                'advantage_type': advantage['advantage_type']
            })
        
        # Generate conclusion
        primary_count = len(explanation['primary_factors'])
        secondary_count = len(explanation['secondary_factors'])
        strategic_count = len(explanation['strategic_elements'])
        
        explanation['conclusion'] = f"""
        This villain achieves unbeatable status through a perfect combination of:
        
        1. {primary_count} extreme advantages in key combat/mental attributes
        2. {secondary_count} significant advantages in supporting characteristics  
        3. {strategic_count} unique strategic elements that provide tactical superiority
        
        The combination of these factors creates a character that excels in all areas
        necessary for victory while maintaining strategic advantages that opponents
        cannot easily counter.
        """
        
        print(f"✅ Explanation generated:")
        print(f"   Primary factors: {primary_count}")
        print(f"   Secondary factors: {secondary_count}")
        print(f"   Strategic elements: {strategic_count}")
        
        return explanation
    
    def create_task3_visualizations(self, save_path: str = 'task3_analysis.png'):
        """
        Create comprehensive visualizations for Task 3 analysis
        
        Args:
            save_path (str): Path to save visualizations
        """
        print(f"\n🎨 Creating Task 3 visualizations...")
        
        if self.feature_comparison is None:
            self.analyze_villain_features()
        
        # Set up the figure
        fig = plt.figure(figsize=(20, 16))
        
        villain = self.task3_villain.iloc[0]
        villain_name = villain.get('name', 'Unknown Villain')
        
        # 1. Numerical features comparison (radar chart)
        plt.subplot(2, 3, 1)
        
        # Select top numerical features for radar chart
        numerical_features = []
        villain_values = []
        dataset_means = []
        
        for feature, comparison in list(self.feature_comparison['numerical_comparison'].items())[:6]:
            numerical_features.append(feature.replace('_', ' ').title())
            # Normalize values for radar chart (0-1 scale)
            villain_norm = comparison['percentile'] / 100
            dataset_norm = 0.5  # Mean is always at 50th percentile
            
            villain_values.append(villain_norm)
            dataset_means.append(dataset_norm)
        
        if len(numerical_features) > 0:
            # Create radar chart
            angles = np.linspace(0, 2*np.pi, len(numerical_features), endpoint=False).tolist()
            villain_values += villain_values[:1]  # Complete the circle
            dataset_means += dataset_means[:1]
            angles += angles[:1]
            
            ax = plt.subplot(2, 3, 1, projection='polar')
            ax.plot(angles, villain_values, 'o-', linewidth=2, label=villain_name, color='red')
            ax.fill(angles, villain_values, alpha=0.25, color='red')
            ax.plot(angles, dataset_means, 'o-', linewidth=2, label='Dataset Average', color='blue')
            ax.fill(angles, dataset_means, alpha=0.15, color='blue')
            
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(numerical_features, fontsize=10)
            ax.set_ylim(0, 1)
            ax.set_title(f'{villain_name} vs Average\n(Percentile Comparison)', fontweight='bold', pad=20)
            ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        
        # 2. Feature deviations from average
        plt.subplot(2, 3, 2)
        
        features = []
        deviations = []
        colors = []
        
        for feature, comparison in self.feature_comparison['numerical_comparison'].items():
            if abs(comparison['percent_diff_from_mean']) > 5:  # Only show significant differences
                features.append(feature.replace('_', ' ').title()[:15])  # Truncate long names
                deviations.append(comparison['percent_diff_from_mean'])
                colors.append('green' if comparison['percent_diff_from_mean'] > 0 else 'red')
        
        if len(features) > 0:
            # Sort by absolute deviation
            sorted_data = sorted(zip(features, deviations, colors), key=lambda x: abs(x[1]), reverse=True)
            features, deviations, colors = zip(*sorted_data[:10])  # Top 10
            
            bars = plt.barh(range(len(features)), deviations, color=colors, alpha=0.7)
            plt.yticks(range(len(features)), features)
            plt.xlabel('Percentage Difference from Average')
            plt.title('Feature Deviations from Dataset Average', fontweight='bold')
            plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
            
            # Add value labels
            for i, bar in enumerate(bars):
                width = bar.get_width()
                plt.text(width + (2 if width > 0 else -2), bar.get_y() + bar.get_height()/2,
                        f'{width:+.1f}%', ha='left' if width > 0 else 'right', va='center', fontsize=9)
        
        # 3. Categorical features uniqueness
        plt.subplot(2, 3, 3)
        
        cat_features = []
        rarity_scores = []
        
        for feature, analysis in self.feature_comparison['categorical_analysis'].items():
            cat_features.append(feature.replace('_', ' ').title())
            # Convert rarity to score (higher = more unique)
            rarity_map = {'Common': 1, 'Uncommon': 2, 'Rare': 3, 'Very Rare': 4, 'Unique/Unknown': 5}
            rarity_scores.append(rarity_map.get(analysis['rarity'], 1))
        
        if len(cat_features) > 0:
            colors_rarity = ['gold' if score >= 4 else 'orange' if score >= 3 else 'lightblue' for score in rarity_scores]
            bars = plt.bar(range(len(cat_features)), rarity_scores, color=colors_rarity, alpha=0.8)
            plt.xticks(range(len(cat_features)), cat_features, rotation=45, ha='right')
            plt.ylabel('Uniqueness Score')
            plt.title('Categorical Feature Uniqueness', fontweight='bold')
            plt.ylim(0, 5)
            
            # Add rarity labels
            rarity_labels = ['Common', 'Uncommon', 'Rare', 'Very Rare', 'Unique']
            for i, bar in enumerate(bars):
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                        rarity_labels[int(height)-1], ha='center', va='bottom', fontsize=8, rotation=45)
        
        # 4. Win probability context
        plt.subplot(2, 3, 4)
        
        # Get win probability distribution from dataset
        win_probs = self.main_dataset['win_prob'].dropna()
        plt.hist(win_probs, bins=30, alpha=0.7, color='lightblue', edgecolor='black', label='Dataset Distribution')
        
        # Add villain's win probability
        villain_win_prob = villain.get('win_prob', 1.0)
        plt.axvline(x=villain_win_prob, color='red', linestyle='--', linewidth=3, 
                   label=f'{villain_name}: {villain_win_prob}')
        
        # Add percentile information
        percentile = (win_probs <= villain_win_prob).mean() * 100
        plt.axvline(x=win_probs.quantile(0.9), color='orange', linestyle=':', alpha=0.7, label='90th Percentile')
        plt.axvline(x=win_probs.quantile(0.95), color='gold', linestyle=':', alpha=0.7, label='95th Percentile')
        
        plt.xlabel('Win Probability')
        plt.ylabel('Frequency')
        plt.title('Win Probability Distribution\n(Villain in Context)', fontweight='bold')
        plt.legend()
        
        # Add text box with percentile info
        plt.text(0.02, 0.98, f'{villain_name} is at\n{percentile:.1f}th percentile', 
                transform=plt.gca().transAxes, fontsize=10,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                verticalalignment='top')
        
        # 5. Feature importance impact
        plt.subplot(2, 3, 5)
        
        if hasattr(self.model, 'feature_importances_'):
            # Get model's feature importance
            feature_importance = dict(zip(self.preprocessor.selected_features, self.model.feature_importances_))
            sorted_importance = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:10]
            
            features_imp = [f.replace('_scaled', '').replace('_encoded', '').replace('_', ' ').title()[:12] 
                           for f, _ in sorted_importance]
            importances = [imp for _, imp in sorted_importance]
            
            bars = plt.barh(range(len(features_imp)), importances, color='purple', alpha=0.7)
            plt.yticks(range(len(features_imp)), features_imp)
            plt.xlabel('Feature Importance')
            plt.title('Model Feature Importance\n(Top 10 Features)', fontweight='bold')
            
            # Add value labels
            for i, bar in enumerate(bars):
                width = bar.get_width()
                plt.text(width + 0.001, bar.get_y() + bar.get_height()/2,
                        f'{width:.3f}', ha='left', va='center', fontsize=9)
        
        # 6. Summary statistics
        plt.subplot(2, 3, 6)
        plt.axis('off')
        
        # Create summary text
        summary_text = f"UNBEATABLE VILLAIN ANALYSIS\n"
        summary_text += f"=" * 30 + "\n\n"
        summary_text += f"Villain: {villain_name}\n"
        summary_text += f"Win Probability: {villain.get('win_prob', 1.0)}\n\n"
        
        # Add key advantages
        extreme_features = [f for f in self.feature_comparison['extreme_features'] 
                          if abs(f['deviation']) > 20]
        
        summary_text += f"KEY ADVANTAGES:\n"
        for i, feature in enumerate(extreme_features[:5]):
            direction = "↑" if feature['deviation'] > 0 else "↓"
            summary_text += f"• {feature['feature']}: {feature['deviation']:+.1f}% {direction}\n"
        
        # Add strategic elements
        strategic_count = len(self.feature_comparison['strategic_advantages'])
        summary_text += f"\nSTRATEGIC ELEMENTS: {strategic_count}\n"
        
        # Add categorical uniqueness
        unique_cats = [cat for cat, analysis in self.feature_comparison['categorical_analysis'].items()
                      if analysis['rarity'] in ['Very Rare', 'Unique/Unknown']]
        summary_text += f"UNIQUE TRAITS: {len(unique_cats)}\n"
        
        # Model prediction accuracy
        prediction_analysis = self.predict_villain_probability()
        summary_text += f"\nMODEL PREDICTION:\n"
        summary_text += f"Predicted: {prediction_analysis['predicted_probability']:.4f}\n"
        summary_text += f"Error: {prediction_analysis['prediction_error']:.4f}\n"
        
        plt.text(0.05, 0.95, summary_text, transform=plt.gca().transAxes, fontsize=11,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
        
        plt.suptitle(f'Task 3: {villain_name} - Unbeatable Villain Analysis', 
                    fontsize=16, fontweight='bold', y=0.98)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"✅ Task 3 visualizations saved as '{save_path}'")
    
    def generate_task3_report(self) -> str:
        """
        Generate comprehensive Task 3 report
        
        Returns:
            str: Task 3 report
        """
        if self.feature_comparison is None:
            self.analyze_villain_features()
        
        villain = self.task3_villain.iloc[0]
        villain_name = villain.get('name', 'Unknown Villain')
        
        report = f"# Task 3: Unbeatable Villain Analysis - {villain_name}\n\n"
        
        # Basic information
        report += f"## Villain Overview\n\n"
        report += f"- **Name**: {villain_name}\n"
        report += f"- **Role**: {villain.get('role', 'Unknown')}\n"
        report += f"- **Win Probability**: {villain.get('win_prob', 1.0)}\n"
        report += f"- **Status**: Unbeatable (Perfect Win Rate)\n\n"
        
        # Model prediction
        prediction_analysis = self.predict_villain_probability()
        report += f"## Model Analysis\n\n"
        report += f"- **Model Prediction**: {prediction_analysis['predicted_probability']:.4f}\n"
        report += f"- **Prediction Error**: {prediction_analysis['prediction_error']:.4f}\n"
        report += f"- **Model Assessment**: {'Successfully identifies as top-tier' if prediction_analysis['model_captures_unbeatable'] else 'Underestimates true strength'}\n\n"
        
        # Numerical features analysis
        report += f"## Numerical Features Analysis\n\n"
        report += f"### Extreme Advantages (>20% deviation)\n\n"
        
        extreme_numerical = [f for f in self.feature_comparison['extreme_features'] 
                           if f['type'] == 'numerical' and abs(f['deviation']) > 20]
        
        if extreme_numerical:
            for feature in extreme_numerical:
                direction = "higher" if feature['deviation'] > 0 else "lower"
                report += f"- **{feature['feature']}**: {feature['deviation']:+.1f}% {direction} than average\n"
        else:
            report += "- No extreme numerical advantages identified\n"
        
        report += f"\n### Significant Advantages (10-20% deviation)\n\n"
        
        significant_numerical = [f for f, comp in self.feature_comparison['numerical_comparison'].items()
                               if 10 <= abs(comp['percent_diff_from_mean']) <= 20]
        
        if significant_numerical:
            for feature in significant_numerical:
                comp = self.feature_comparison['numerical_comparison'][feature]
                direction = "higher" if comp['percent_diff_from_mean'] > 0 else "lower"
                report += f"- **{feature}**: {comp['percent_diff_from_mean']:+.1f}% {direction} than average ({comp['percentile']:.1f}th percentile)\n"
        else:
            report += "- No significant numerical advantages identified\n"
        
        # Categorical features analysis
        report += f"\n## Categorical Features Analysis\n\n"
        
        for feature, analysis in self.feature_comparison['categorical_analysis'].items():
            report += f"### {feature}\n"
            report += f"- **Villain's Value**: {analysis['villain_value']}\n"
            report += f"- **Frequency in Dataset**: {analysis['frequency_count']} characters ({analysis['frequency_percentage']:.1f}%)\n"
            report += f"- **Rarity Level**: {analysis['rarity']}\n"
            report += f"- **Most Common Value**: {analysis['most_common']}\n"
            
            if analysis['is_mode']:
                report += f"- **Strategic Advantage**: Uses most popular/proven effective value\n"
            elif analysis['rarity'] in ['Very Rare', 'Unique/Unknown']:
                report += f"- **Strategic Advantage**: Unique trait that opponents rarely encounter\n"
            
            report += "\n"
        
        # Feature importance analysis
        if hasattr(self.model, 'feature_importances_'):
            importance_analysis = self.analyze_feature_importance_impact()
            
            report += f"## Feature Importance Impact\n\n"
            report += f"- **Villain's Advantage Score**: {importance_analysis['villain_advantage_score']:.4f}\n"
            report += f"- **Key Strengths**: {len(importance_analysis['key_strengths'])}\n\n"
            
            if importance_analysis['key_strengths']:
                report += f"### Top Strengths (High Importance + High Performance)\n\n"
                for strength in importance_analysis['key_strengths']:
                    report += f"- **{strength['feature']}**: Importance {strength['importance']:.4f}, Performance {strength['performance']:.4f}\n"
        
        # Explanation
        explanation = self.explain_unbeatable_status()
        
        report += f"\n## Why This Villain Is Unbeatable\n\n"
        
        if explanation['primary_factors']:
            report += f"### Primary Factors (Extreme Advantages)\n"
            for factor in explanation['primary_factors']:
                report += f"- **{factor['factor']}**: {factor['description']} ({factor['deviation']:+.1f}% from average)\n"
            report += "\n"
        
        if explanation['secondary_factors']:
            report += f"### Secondary Factors (Significant Advantages)\n"
            for factor in explanation['secondary_factors']:
                report += f"- **{factor['factor']}**: {factor['deviation']:+.1f}% from average ({factor['percentile']:.1f}th percentile)\n"
            report += "\n"
        
        if explanation['strategic_elements']:
            report += f"### Strategic Elements\n"
            for element in explanation['strategic_elements']:
                if element['advantage_type'] == 'unique':
                    report += f"- **{element['factor']}**: Unique/rare value '{element['value']}' provides tactical surprise\n"
                else:
                    report += f"- **{element['factor']}**: Popular value '{element['value']}' represents proven effectiveness\n"
            report += "\n"
        
        # Conclusion
        report += f"## Conclusion\n\n"
        report += explanation['conclusion'].strip()
        
        # Dataset context
        dataset_stats = self.calculate_dataset_statistics()
        win_prob_data = self.main_dataset['win_prob'].dropna()
        villain_percentile = (win_prob_data <= villain.get('win_prob', 1.0)).mean() * 100
        
        report += f"\n\n## Dataset Context\n\n"
        report += f"- **Dataset Size**: {len(self.main_dataset):,} characters\n"
        report += f"- **Villain's Percentile**: {villain_percentile:.1f}th percentile\n"
        report += f"- **Characters with >90% win rate**: {dataset_stats['win_prob_stats']['high_performers_count']}\n"
        report += f"- **Perfect performers**: {dataset_stats['win_prob_stats']['perfect_performers_count']}\n"
        
        return report
    
    def run_complete_task3_analysis(self) -> Dict:
        """
        Run complete Task 3 analysis pipeline
        
        Returns:
            Dict: Complete Task 3 results
        """
        print("🦹 Starting Complete Task 3 Analysis")
        print("=" * 80)
        
        # Step 1: Analyze villain features
        feature_analysis = self.analyze_villain_features()
        
        # Step 2: Predict villain probability
        prediction_analysis = self.predict_villain_probability()
        
        # Step 3: Analyze feature importance impact
        importance_analysis = self.analyze_feature_importance_impact()
        
        # Step 4: Generate explanation
        explanation = self.explain_unbeatable_status()
        
        # Step 5: Create visualizations
        self.create_task3_visualizations()
        
        # Step 6: Generate report
        report = self.generate_task3_report()
        with open('task3_report.md', 'w', encoding='utf-8') as f:
            f.write(report)
        print("📄 Task 3 report saved as 'task3_report.md'")
        
        print("\n" + "🎉" * 20)
        print("TASK 3 ANALYSIS COMPLETED SUCCESSFULLY!")
        print("🎉" * 20)
        
        results = {
            'feature_analysis': feature_analysis,
            'prediction_analysis': prediction_analysis,
            'importance_analysis': importance_analysis,
            'explanation': explanation,
            'report': report
        }
        
        # Print summary
        villain = self.task3_villain.iloc[0]
        villain_name = villain.get('name', 'Unknown')
        
        print(f"\n📊 Task 3 Summary:")
        print(f"   Villain analyzed: {villain_name}")
        print(f"   Extreme advantages: {len([f for f in feature_analysis['extreme_features'] if abs(f['deviation']) > 20])}")
        print(f"   Strategic elements: {len(feature_analysis['strategic_advantages'])}")
        print(f"   Model prediction: {prediction_analysis['predicted_probability']:.4f}")
        print(f"   Prediction accuracy: {abs(prediction_analysis['prediction_error']):.4f} error")
        
        # Key insights
        print(f"\n🎯 Key Insights:")
        extreme_features = [f for f in feature_analysis['extreme_features'] if abs(f['deviation']) > 30]
        if extreme_features:
            top_advantage = max(extreme_features, key=lambda x: abs(x['deviation']))
            print(f"   Biggest advantage: {top_advantage['feature']} ({top_advantage['deviation']:+.1f}%)")
        
        unique_traits = [cat for cat, analysis in feature_analysis['categorical_analysis'].items()
                        if analysis['rarity'] in ['Very Rare', 'Unique/Unknown']]
        if unique_traits:
            print(f"   Unique traits: {len(unique_traits)} rare/unique categorical features")
        
        if importance_analysis and 'villain_advantage_score' in importance_analysis:
            print(f"   Overall advantage score: {importance_analysis['villain_advantage_score']:.3f}")
        
        return results

# Example usage
if __name__ == "__main__":
    from dataloader import DataLoader
    from preprocessing import DataPreprocessor
    from model_performance import ModelPerformanceAnalyzer
    
    # Load data
    loader = DataLoader()
    main_data, _, _, task3_villain = loader.load_all_data()
    
    if all(data is not None for data in [main_data, task3_villain]):
        # Preprocess data and train model
        preprocessor = DataPreprocessor()
        X_train, X_val, X_test, y_train, y_val, y_test, _ = preprocessor.run_complete_preprocessing(main_data)
        
        # Train model
        analyzer = ModelPerformanceAnalyzer()
        analyzer.train_models(X_train, y_train, X_val, y_val)
        
        # Run Task 3 analysis
        task3_analyzer = Task3VillainAnalyzer(analyzer.best_model, preprocessor, main_data)
        task3_analyzer.load_task3_data(task3_villain)
        results = task3_analyzer.run_complete_task3_analysis()
        
        print(f"\n🦹 Task 3 Complete!")
    else:
        print("❌ Failed to load required data for Task 3 analysis")