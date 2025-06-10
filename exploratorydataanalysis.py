"""
Exploratory Data Analysis (EDA) Module
Comprehensive analysis of superhero/villain dataset with visualizations
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

class SuperheroEDA:
    """
    Class for performing comprehensive Exploratory Data Analysis
    """
    
    def __init__(self, data: pd.DataFrame):
        """
        Initialize EDA with dataset
        
        Args:
            data (pd.DataFrame): Main dataset
        """
        self.data = data.copy()
        self.numerical_features = [
            'power_level', 'weight', 'height', 'age', 'gender', 'speed', 
            'battle_iq', 'ranking', 'intelligence', 'training_time', 'secret_code'
        ]
        self.categorical_features = [
            'role', 'skin_type', 'eye_color', 'hair_color', 'universe', 
            'body_type', 'job', 'species', 'abilities', 'special_attack'
        ]
        self.target = 'win_prob'
        
        # Filter features that exist in the dataset
        self.numerical_features = [f for f in self.numerical_features if f in self.data.columns]
        self.categorical_features = [f for f in self.categorical_features if f in self.data.columns]
        
        # Set plotting style
        plt.style.use('seaborn-v0_8-whitegrid')
        sns.set_palette('Set2')
        
        # Results storage
        self.eda_results = {}
    
    def basic_info(self) -> Dict:
        """
        Get basic information about the dataset
        
        Returns:
            Dict: Basic dataset information
        """
        print("=" * 60)
        print("BASIC DATASET INFORMATION")
        print("=" * 60)
        
        info = {
            'shape': self.data.shape,
            'memory_usage_mb': self.data.memory_usage(deep=True).sum() / 1024**2,
            'dtypes': self.data.dtypes.value_counts().to_dict(),
            'missing_values': self.data.isnull().sum().sum(),
            'duplicate_rows': self.data.duplicated().sum()
        }
        
        print(f" Dataset Shape: {info['shape'][0]} rows × {info['shape'][1]} columns")
        print(f" Memory Usage: {info['memory_usage_mb']:.2f} MB")
        print(f" Data Types: {info['dtypes']}")
        print(f" Missing Values: {info['missing_values']}")
        print(f" Duplicate Rows: {info['duplicate_rows']}")
        
        self.eda_results['basic_info'] = info
        return info
    
    def missing_values_analysis(self) -> pd.DataFrame:
        """
        Analyze missing values in the dataset
        
        Returns:
            pd.DataFrame: Missing values summary
        """
        print("\n" + "=" * 60)
        print("MISSING VALUES ANALYSIS")
        print("=" * 60)
        
        missing_data = self.data.isnull().sum()
        missing_percent = (missing_data / len(self.data)) * 100
        
        missing_df = pd.DataFrame({
            'Column': missing_data.index,
            'Missing_Count': missing_data.values,
            'Missing_Percentage': missing_percent.values
        }).sort_values('Missing_Percentage', ascending=False)
        
        # Filter columns with missing values
        missing_df = missing_df[missing_df['Missing_Count'] > 0]
        
        if len(missing_df) > 0:
            print(" Features with missing values:")
            for _, row in missing_df.iterrows():
                print(f"   {row['Column']}: {row['Missing_Count']} ({row['Missing_Percentage']:.2f}%)")
                
            # Categorize missing value severity
            high_missing = missing_df[missing_df['Missing_Percentage'] > 30]
            medium_missing = missing_df[(missing_df['Missing_Percentage'] > 10) & 
                                      (missing_df['Missing_Percentage'] <= 30)]
            low_missing = missing_df[missing_df['Missing_Percentage'] <= 10]
            
            print(f"\n Missing Value Categories:")
            print(f"   🔴 High (>30%): {len(high_missing)} features")
            print(f"   🟡 Medium (10-30%): {len(medium_missing)} features")
            print(f"   🟢 Low (≤10%): {len(low_missing)} features")
        else:
            print(" No missing values found!")
        
        self.eda_results['missing_values'] = missing_df
        return missing_df
    
    def target_variable_analysis(self) -> Dict:
        """
        Analyze the target variable (win_prob)
        
        Returns:
            Dict: Target variable statistics
        """
        print("\n" + "=" * 60)
        print("TARGET VARIABLE ANALYSIS (WIN_PROB)")
        print("=" * 60)
        
        target_stats = self.data[self.target].describe()
        
        # Additional statistics
        q1 = self.data[self.target].quantile(0.25)
        q3 = self.data[self.target].quantile(0.75)
        iqr = q3 - q1
        
        # Outlier detection
        outlier_lower = q1 - 1.5 * iqr
        outlier_upper = q3 + 1.5 * iqr
        outliers = self.data[(self.data[self.target] < outlier_lower) | 
                           (self.data[self.target] > outlier_upper)]
        
        analysis = {
            'statistics': target_stats.to_dict(),
            'iqr': iqr,
            'outliers_count': len(outliers),
            'outliers_percentage': len(outliers) / len(self.data) * 100,
            'skewness': self.data[self.target].skew(),
            'kurtosis': self.data[self.target].kurtosis()
        }
        
        print(" Descriptive Statistics:")
        for stat, value in target_stats.items():
            print(f"   {stat.capitalize()}: {value:.4f}")
        
        print(f"\n Distribution Properties:")
        print(f"   IQR: {iqr:.4f}")
        print(f"   Skewness: {analysis['skewness']:.4f}")
        print(f"   Kurtosis: {analysis['kurtosis']:.4f}")
        print(f"   Outliers: {analysis['outliers_count']} ({analysis['outliers_percentage']:.2f}%)")
        
        if analysis['outliers_count'] > 0:
            extreme_outliers = outliers.nlargest(5, self.target)
            print(f"   Top 5 outliers: {extreme_outliers[self.target].values}")
        
        self.eda_results['target_analysis'] = analysis
        return analysis
    
    def role_analysis(self) -> Dict:
        """
        Analyze performance by character role
        
        Returns:
            Dict: Role analysis results
        """
        print("\n" + "=" * 60)
        print("ROLE PERFORMANCE ANALYSIS")
        print("=" * 60)
        
        role_stats = self.data.groupby('role')[self.target].agg([
            'count', 'mean', 'median', 'std', 'min', 'max'
        ]).round(4)
        
        role_stats = role_stats.sort_values('mean', ascending=False)
        
        print(" Win Probability by Role:")
        print(role_stats.to_string())
        
        # Hero vs Villain analysis
        hero_roles = [role for role in self.data['role'].unique() 
                     if role and 'hero' in str(role).lower()]
        villain_roles = [role for role in self.data['role'].unique() 
                        if role and 'villain' in str(role).lower()]
        
        hero_data = self.data[self.data['role'].isin(hero_roles)]
        villain_data = self.data[self.data['role'].isin(villain_roles)]
        
        comparison = {}
        if len(hero_data) > 0 and len(villain_data) > 0:
            hero_avg = hero_data[self.target].mean()
            villain_avg = villain_data[self.target].mean()
            advantage = ((villain_avg - hero_avg) / hero_avg) * 100
            
            comparison = {
                'hero_avg': hero_avg,
                'villain_avg': villain_avg,
                'villain_advantage_pct': advantage,
                'hero_count': len(hero_data),
                'villain_count': len(villain_data)
            }
            
            print(f"\n🦸 Hero vs Villain Comparison:")
            print(f"   Heroes average: {hero_avg:.4f} ({comparison['hero_count']} characters)")
            print(f"   Villains average: {villain_avg:.4f} ({comparison['villain_count']} characters)")
            print(f"   Villain advantage: {advantage:.2f}%")
        
        analysis = {
            'role_stats': role_stats,
            'hero_villain_comparison': comparison
        }
        
        self.eda_results['role_analysis'] = analysis
        return analysis
    
    def numerical_features_analysis(self) -> Dict:
        """
        Analyze numerical features and their correlation with target
        
        Returns:
            Dict: Numerical features analysis
        """
        print("\n" + "=" * 60)
        print("NUMERICAL FEATURES ANALYSIS")
        print("=" * 60)
        
        # Correlation analysis
        correlations = {}
        for feature in self.numerical_features:
            if feature in self.data.columns:
                corr = self.data[feature].corr(self.data[self.target])
                if pd.notnull(corr):
                    correlations[feature] = corr
        
        # Sort by absolute correlation
        sorted_correlations = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)
        
        print("🔗 Correlation with win_prob (sorted by absolute correlation):")
        for feature, corr in sorted_correlations:
            direction = "positive" if corr > 0 else "negative"
            strength = "strong" if abs(corr) > 0.5 else "moderate" if abs(corr) > 0.3 else "weak"
            print(f"   {feature}: {corr:.4f} ({direction}, {strength})")
        
        # Outlier analysis
        print(f"\n Outlier Analysis:")
        outlier_summary = {}
        
        for feature in self.numerical_features:
            if feature in self.data.columns and self.data[feature].notnull().sum() > 0:
                q1 = self.data[feature].quantile(0.25)
                q3 = self.data[feature].quantile(0.75)
                iqr = q3 - q1
                
                outlier_lower = q1 - 1.5 * iqr
                outlier_upper = q3 + 1.5 * iqr
                
                outliers = self.data[(self.data[feature] < outlier_lower) | 
                                   (self.data[feature] > outlier_upper)]
                
                outlier_pct = len(outliers) / len(self.data) * 100
                
                outlier_summary[feature] = {
                    'count': len(outliers),
                    'percentage': outlier_pct,
                    'lower_bound': outlier_lower,
                    'upper_bound': outlier_upper
                }
                
                if outlier_pct > 0:
                    print(f"   {feature}: {len(outliers)} outliers ({outlier_pct:.2f}%)")
        
        analysis = {
            'correlations': correlations,
            'sorted_correlations': sorted_correlations,
            'outlier_summary': outlier_summary
        }
        
        self.eda_results['numerical_analysis'] = analysis
        return analysis
    
    def categorical_features_analysis(self) -> Dict:
        """
        Analyze categorical features
        
        Returns:
            Dict: Categorical features analysis
        """
        print("\n" + "=" * 60)
        print("CATEGORICAL FEATURES ANALYSIS")
        print("=" * 60)
        
        categorical_summary = {}
        
        for feature in self.categorical_features:
            if feature in self.data.columns:
                unique_count = self.data[feature].nunique()
                value_counts = self.data[feature].value_counts()
                
                # Calculate win probability by category (if reasonable number of categories)
                category_performance = None
                if 1 < unique_count <= 20:  # Reasonable number for analysis
                    category_performance = self.data.groupby(feature)[self.target].agg([
                        'count', 'mean'
                    ]).sort_values('mean', ascending=False)
                
                categorical_summary[feature] = {
                    'unique_count': unique_count,
                    'top_values': value_counts.head(3).to_dict(),
                    'category_performance': category_performance
                }
                
                print(f"\n {feature}:")
                print(f"   Unique values: {unique_count}")
                print(f"   Top 3: {list(value_counts.head(3).index)}")
                
                if category_performance is not None:
                    best_category = category_performance.index[0]
                    worst_category = category_performance.index[-1]
                    best_performance = category_performance.loc[best_category, 'mean']
                    worst_performance = category_performance.loc[worst_category, 'mean']
                    
                    print(f"   Best performing: {best_category} ({best_performance:.4f})")
                    print(f"   Worst performing: {worst_category} ({worst_performance:.4f})")
        
        self.eda_results['categorical_analysis'] = categorical_summary
        return categorical_summary
    
    def high_performers_analysis(self) -> Dict:
        """
        Analyze characteristics of high-performing characters
        
        Returns:
            Dict: High performers analysis
        """
        print("\n" + "=" * 60)
        print("HIGH PERFORMERS ANALYSIS")
        print("=" * 60)
        
        # Define high performers (top 10% or win_prob > 0.9)
        high_threshold = max(0.9, self.data[self.target].quantile(0.9))
        high_performers = self.data[self.data[self.target] > high_threshold]
        
        print(f" High performers (win_prob > {high_threshold:.3f}): {len(high_performers)} characters ({len(high_performers)/len(self.data)*100:.2f}%)")
        
        if len(high_performers) > 0:
            # Top performers
            top_performers = high_performers.nlargest(5, self.target)
            print(f"\n🥇 Top 5 performers:")
            for _, char in top_performers.iterrows():
                print(f"   {char.get('name', 'Unknown')} ({char.get('role', 'Unknown')}): {char[self.target]:.4f}")
            
            # Characteristics comparison
            print(f"\n High performers vs. average (numerical features):")
            numerical_comparison = {}
            
            for feature in self.numerical_features:
                if feature in self.data.columns and self.data[feature].notnull().sum() > 0:
                    high_avg = high_performers[feature].mean()
                    overall_avg = self.data[feature].mean()
                    
                    if pd.notnull(high_avg) and pd.notnull(overall_avg) and overall_avg != 0:
                        pct_diff = ((high_avg - overall_avg) / overall_avg) * 100
                        
                        numerical_comparison[feature] = {
                            'high_avg': high_avg,
                            'overall_avg': overall_avg,
                            'pct_diff': pct_diff
                        }
                        
                        if abs(pct_diff) > 5:  # Only show significant differences
                            direction = "higher" if pct_diff > 0 else "lower"
                            print(f"   {feature}: {pct_diff:+.2f}% {direction} than average")
            
            # Categorical preferences
            print(f"\n Common characteristics (categorical features):")
            categorical_preferences = {}
            
            for feature in self.categorical_features:
                if feature in self.data.columns:
                    high_perf_counts = high_performers[feature].value_counts()
                    if len(high_perf_counts) > 0:
                        most_common = high_perf_counts.index[0]
                        count = high_perf_counts.iloc[0]
                        percentage = (count / len(high_performers)) * 100
                        
                        categorical_preferences[feature] = {
                            'most_common': most_common,
                            'count': count,
                            'percentage': percentage
                        }
                        
                        print(f"   {feature}: {most_common} ({count}/{len(high_performers)}, {percentage:.1f}%)")
        
        analysis = {
            'threshold': high_threshold,
            'count': len(high_performers),
            'percentage': len(high_performers)/len(self.data)*100,
            'top_performers': top_performers if len(high_performers) > 0 else pd.DataFrame(),
            'numerical_comparison': numerical_comparison if len(high_performers) > 0 else {},
            'categorical_preferences': categorical_preferences if len(high_performers) > 0 else {}
        }
        
        self.eda_results['high_performers'] = analysis
        return analysis
    
    def create_visualizations(self, save_path: str = 'eda_visualizations.png'):
        """
        Create comprehensive EDA visualizations
        
        Args:
            save_path (str): Path to save the visualization
        """
        print(f"\n Creating EDA visualizations...")
        
        # Set up the figure with subplots
        fig = plt.figure(figsize=(20, 16))
        
        # 1. Target variable distribution
        plt.subplot(3, 3, 1)
        sns.histplot(data=self.data, x=self.target, bins=30, kde=True)
        plt.title('Win Probability Distribution', fontsize=12, fontweight='bold')
        plt.xlabel('Win Probability')
        plt.ylabel('Frequency')
        
        # 2. Win probability by role
        plt.subplot(3, 3, 2)
        if 'role' in self.data.columns:
            role_means = self.data.groupby('role')[self.target].mean().sort_values(ascending=False)
            colors = ['red' if 'villain' in role.lower() else 'blue' if 'hero' in role.lower() else 'gray' 
                     for role in role_means.index]
            bars = plt.bar(range(len(role_means)), role_means.values, color=colors)
            plt.title('Average Win Probability by Role', fontsize=12, fontweight='bold')
            plt.xlabel('Role')
            plt.ylabel('Average Win Probability')
            plt.xticks(range(len(role_means)), role_means.index, rotation=45, ha='right')
            
            # Add value labels on bars
            for i, bar in enumerate(bars):
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{height:.3f}', ha='center', va='bottom', fontsize=9)
        
        # 3. Correlation heatmap
        plt.subplot(3, 3, 3)
        corr_features = [self.target] + self.numerical_features[:6]  # Limit to 6 for readability
        corr_features = [f for f in corr_features if f in self.data.columns]
        
        if len(corr_features) > 1:
            corr_matrix = self.data[corr_features].corr()
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
            sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', center=0, 
                       fmt='.2f', square=True)
            plt.title('Feature Correlation Matrix', fontsize=12, fontweight='bold')
        
        # 4. Box plot of win_prob by role
        plt.subplot(3, 3, 4)
        if 'role' in self.data.columns:
            sns.boxplot(data=self.data, x='role', y=self.target)
            plt.title('Win Probability Distribution by Role', fontsize=12, fontweight='bold')
            plt.xlabel('Role')
            plt.ylabel('Win Probability')
            plt.xticks(rotation=45, ha='right')
        
        # 5. Scatter plot: power_level vs win_prob
        plt.subplot(3, 3, 5)
        if 'power_level' in self.data.columns:
            plt.scatter(self.data['power_level'], self.data[self.target], alpha=0.5, s=20)
            plt.xlabel('Power Level')
            plt.ylabel('Win Probability')
            plt.title('Power Level vs Win Probability', fontsize=12, fontweight='bold')
        
        # 6. Top species performance
        plt.subplot(3, 3, 6)
        if 'species' in self.data.columns:
            species_perf = self.data.groupby('species')[self.target].mean().sort_values(ascending=False).head(8)
            plt.barh(range(len(species_perf)), species_perf.values)
            plt.yticks(range(len(species_perf)), species_perf.index)
            plt.title('Top Species by Win Probability', fontsize=12, fontweight='bold')
            plt.xlabel('Average Win Probability')
        
        # 7. Battle IQ vs Intelligence
        plt.subplot(3, 3, 7)
        if 'battle_iq' in self.data.columns and 'intelligence' in self.data.columns:
            scatter = plt.scatter(self.data['battle_iq'], self.data['intelligence'], 
                       c=self.data[self.target], cmap='viridis', alpha=0.6, s=20)
            plt.colorbar(scatter, label='Win Probability')
            plt.xlabel('Battle IQ')
            plt.ylabel('Intelligence')
            plt.title('Battle IQ vs Intelligence\n(colored by Win Prob)', fontsize=12, fontweight='bold')
        
        # 8. Missing values heatmap
        plt.subplot(3, 3, 8)
        missing_data = self.data.isnull()
        if missing_data.sum().sum() > 0:
            # Show only columns with missing values
            missing_cols = missing_data.columns[missing_data.sum() > 0]
            if len(missing_cols) > 0:
                sns.heatmap(missing_data[missing_cols], cbar=True, yticklabels=False, 
                           cmap='viridis', cbar_kws={'label': 'Missing'})
                plt.title('Missing Values Pattern', fontsize=12, fontweight='bold')
                plt.xlabel('Features')
        else:
            plt.text(0.5, 0.5, 'No Missing Values', ha='center', va='center', fontsize=14)
            plt.title('Missing Values Pattern', fontsize=12, fontweight='bold')
        
        # 9. Universe distribution
        plt.subplot(3, 3, 9)
        if 'universe' in self.data.columns:
            universe_counts = self.data['universe'].value_counts().head(6)
            plt.pie(universe_counts.values, labels=universe_counts.index, autopct='%1.1f%%')
            plt.title('Character Distribution by Universe', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f" EDA visualizations saved as '{save_path}'")
    
    def generate_eda_report(self) -> str:
        """
        Generate a comprehensive EDA report
        
        Returns:
            str: EDA report
        """
        report = "# Exploratory Data Analysis Report\n\n"
        
        # Basic info
        if 'basic_info' in self.eda_results:
            info = self.eda_results['basic_info']
            report += f"## Dataset Overview\n"
            report += f"- **Rows**: {info['shape'][0]:,}\n"
            report += f"- **Columns**: {info['shape'][1]}\n"
            report += f"- **Memory Usage**: {info['memory_usage_mb']:.2f} MB\n"
            report += f"- **Missing Values**: {info['missing_values']:,}\n"
            report += f"- **Duplicate Rows**: {info['duplicate_rows']:,}\n\n"
        
        # Target variable
        if 'target_analysis' in self.eda_results:
            target = self.eda_results['target_analysis']
            report += f"## Target Variable Analysis (win_prob)\n"
            report += f"- **Mean**: {target['statistics']['mean']:.4f}\n"
            report += f"- **Median**: {target['statistics']['50%']:.4f}\n"
            report += f"- **Standard Deviation**: {target['statistics']['std']:.4f}\n"
            report += f"- **Range**: {target['statistics']['min']:.4f} - {target['statistics']['max']:.4f}\n"
            report += f"- **Outliers**: {target['outliers_count']} ({target['outliers_percentage']:.2f}%)\n"
            report += f"- **Skewness**: {target['skewness']:.4f}\n\n"
        
        # Role analysis
        if 'role_analysis' in self.eda_results:
            role = self.eda_results['role_analysis']
            report += f"## Role Performance Analysis\n"
            if 'hero_villain_comparison' in role and role['hero_villain_comparison']:
                comp = role['hero_villain_comparison']
                report += f"- **Heroes Average**: {comp['hero_avg']:.4f} ({comp['hero_count']} characters)\n"
                report += f"- **Villains Average**: {comp['villain_avg']:.4f} ({comp['villain_count']} characters)\n"
                report += f"- **Villain Advantage**: {comp['villain_advantage_pct']:.2f}%\n\n"
        
        # Numerical features
        if 'numerical_analysis' in self.eda_results:
            numerical = self.eda_results['numerical_analysis']
            report += f"## Key Correlations with Win Probability\n"
            for feature, corr in numerical['sorted_correlations'][:5]:
                report += f"- **{feature}**: {corr:.4f}\n"
            report += "\n"
        
        # High performers
        if 'high_performers' in self.eda_results:
            high_perf = self.eda_results['high_performers']
            report += f"## High Performers Analysis\n"
            report += f"- **Count**: {high_perf['count']} ({high_perf['percentage']:.2f}%)\n"
            report += f"- **Threshold**: {high_perf['threshold']:.4f}\n"
            
            if not high_perf['top_performers'].empty:
                report += f"\n**Top 3 Performers**:\n"
                for i, (_, char) in enumerate(high_perf['top_performers'].head(3).iterrows()):
                    name = char.get('name', f'Character_{i+1}')
                    role = char.get('role', 'Unknown')
                    win_prob = char[self.target]
                    report += f"{i+1}. {name} ({role}): {win_prob:.4f}\n"
        
        return report
    
    def run_complete_eda(self, save_viz: bool = True, save_report: bool = True) -> Dict:
        """
        Run complete EDA pipeline
        
        Args:
            save_viz (bool): Whether to save visualizations
            save_report (bool): Whether to save report
            
        Returns:
            Dict: Complete EDA results
        """
        print(" Starting Comprehensive EDA...")
        print("=" * 80)
        
        # Run all analyses
        self.basic_info()
        self.missing_values_analysis()
        self.target_variable_analysis()
        self.role_analysis()
        self.numerical_features_analysis()
        self.categorical_features_analysis()
        self.high_performers_analysis()
        
        # Create visualizations
        if save_viz:
            self.create_visualizations()
        
        # Generate report
        if save_report:
            report = self.generate_eda_report()
            with open('eda_report.md', 'w') as f:
                f.write(report)
            print(" EDA report saved as 'eda_report.md'")
        
        print("\n EDA completed successfully!")
        
        return self.eda_results

# Example usage
if __name__ == "__main__":
    from dataloader import DataLoader
    
    # Load data
    loader = DataLoader()
    main_data, _, _, _ = loader.load_all_data()
    
    if main_data is not None:
        # Run EDA
        eda = SuperheroEDA(main_data)
        results = eda.run_complete_eda()
        
        print("\n🎯 EDA Summary:")
        print(f"- Dataset size: {main_data.shape}")
        print(f"- Analysis components: {len(results)}")
        print(f"- Visualizations: Created")
        print(f"- Report: Generated")
    else:
        print("❌ Failed to load data for EDA")