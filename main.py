#!/usr/bin/env python3
"""
Main Analysis Pipeline
Comprehensive superhero and villain win probability analysis system

This script orchestrates the complete analysis pipeline including:
- Data loading and validation
- Exploratory Data Analysis (EDA)
- Data preprocessing
- Model training and performance evaluation
- Task 2: Match predictions
- Task 3: Unbeatable villain analysis
- Report generation and visualization

Author: Group 
Date: 22-06-2025
"""

import sys
import os
import argparse
import warnings
from pathlib import Path
from datetime import datetime

# Import custom modules
from dataloader import DataLoader
from exploratorydataanalysis import SuperheroEDA
from preprocessing import DataPreprocessor
from model_performance import ModelPerformanceAnalyzer
from task2 import Task2MatchPredictor
from task3 import Task3VillainAnalyzer

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

class SuperheroAnalysisPipeline:
    """
    Main pipeline orchestrator for superhero/villain analysis
    """
    
    def __init__(self):
        """Initialize the analysis pipeline"""
        self.loader = DataLoader()
        self.eda_analyzer = None
        self.preprocessor = DataPreprocessor()
        self.model_analyzer = ModelPerformanceAnalyzer()
        self.task2_predictor = None
        self.task3_analyzer = None
        
        # Data storage
        self.main_data = None
        self.task2_characters = None
        self.task2_matches = None
        self.task3_villain = None
        
        # Results storage
        self.results = {
            'eda': None,
            'preprocessing': None,
            'model_performance': None,
            'task2': None,
            'task3': None
        }
        
        # Pipeline configuration
        self.config = {
            'save_results': True,
            'create_visualizations': True,
            'generate_reports': True,
            'run_all_tasks': True
        }
    
    def setup_environment(self):
        """Setup the analysis environment"""
        print("🚀 SUPERHERO & VILLAIN WIN PROBABILITY ANALYSIS")
        print("=" * 80)
        print(f"Analysis started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Working directory: {os.getcwd()}")
        
        # Create output directories if they don't exist
        output_dirs = ['reports', 'visualizations', 'models']
        for dir_name in output_dirs:
            Path(dir_name).mkdir(exist_ok=True)
        
        print("✅ Environment setup complete")
    
    def load_data(self) -> bool:
        """
        Load all required datasets
        
        Returns:
            bool: True if all data loaded successfully
        """
        print(f"\n" + "=" * 60)
        print("STEP 1: DATA LOADING")
        print("=" * 60)
        
        # Load all datasets
        self.main_data, self.task2_characters, self.task2_matches, self.task3_villain = \
            self.loader.load_all_data()
        
        # Validate data loading
        if all(data is not None for data in [self.main_data, self.task2_characters, 
                                           self.task2_matches, self.task3_villain]):
            print("✅ All datasets loaded successfully!")
            return True
        else:
            print("❌ Failed to load required datasets!")
            return False
    
    def run_eda(self) -> bool:
        """
        Run exploratory data analysis
        
        Returns:
            bool: True if EDA completed successfully
        """
        print(f"\n" + "=" * 60)
        print("STEP 2: EXPLORATORY DATA ANALYSIS")
        print("=" * 60)
        
        try:
            self.eda_analyzer = SuperheroEDA(self.main_data)
            self.results['eda'] = self.eda_analyzer.run_complete_eda(
                save_viz=self.config['create_visualizations'],
                save_report=self.config['generate_reports']
            )
            print("✅ EDA completed successfully!")
            return True
        except Exception as e:
            print(f"❌ EDA failed: {str(e)}")
            return False
    
    def run_preprocessing(self) -> bool:
        """
        Run data preprocessing
        
        Returns:
            bool: True if preprocessing completed successfully
        """
        print(f"\n" + "=" * 60)
        print("STEP 3: DATA PREPROCESSING")
        print("=" * 60)
        
        try:
            self.X_train, self.X_val, self.X_test, self.y_train, self.y_val, self.y_test, processed_data = \
                self.preprocessor.run_complete_preprocessing(
                    self.main_data, 
                    save_pipeline=self.config['save_results']
                )
            
            self.results['preprocessing'] = {
                'processed_data': processed_data,
                'train_shape': self.X_train.shape,
                'val_shape': self.X_val.shape,
                'test_shape': self.X_test.shape,
                'selected_features': self.preprocessor.selected_features
            }
            
            print("✅ Data preprocessing completed successfully!")
            return True
        except Exception as e:
            print(f"❌ Preprocessing failed: {str(e)}")
            return False
    
    def run_model_training(self) -> bool:
        """
        Run model training and performance analysis
        
        Returns:
            bool: True if model training completed successfully
        """
        print(f"\n" + "=" * 60)
        print("STEP 4: MODEL TRAINING & PERFORMANCE EVALUATION")
        print("=" * 60)
        
        try:
            self.results['model_performance'] = self.model_analyzer.run_complete_analysis(
                self.X_train, self.y_train,
                self.X_val, self.y_val,
                self.X_test, self.y_test,
                self.preprocessor.selected_features,
                save_results=self.config['save_results']
            )
            
            print("✅ Model training and evaluation completed successfully!")
            return True
        except Exception as e:
            print(f"❌ Model training failed: {str(e)}")
            return False
    
    def run_task2(self) -> bool:
        """
        Run Task 2 match predictions
        
        Returns:
            bool: True if Task 2 completed successfully
        """
        print(f"\n" + "=" * 60)
        print("STEP 5: TASK 2 - MATCH PREDICTIONS")
        print("=" * 60)
        
        try:
            self.task2_predictor = Task2MatchPredictor(
                self.model_analyzer.best_model, 
                self.preprocessor
            )
            self.task2_predictor.load_task2_data(self.task2_characters, self.task2_matches)
            self.results['task2'] = self.task2_predictor.run_complete_task2_analysis()
            
            print("✅ Task 2 completed successfully!")
            return True
        except Exception as e:
            print(f"❌ Task 2 failed: {str(e)}")
            return False
    
    def run_task3(self) -> bool:
        """
        Run Task 3 unbeatable villain analysis
        
        Returns:
            bool: True if Task 3 completed successfully
        """
        print(f"\n" + "=" * 60)
        print("STEP 6: TASK 3 - UNBEATABLE VILLAIN ANALYSIS")
        print("=" * 60)
        
        try:
            self.task3_analyzer = Task3VillainAnalyzer(
                self.model_analyzer.best_model,
                self.preprocessor,
                self.main_data
            )
            self.task3_analyzer.load_task3_data(self.task3_villain)
            self.results['task3'] = self.task3_analyzer.run_complete_task3_analysis()
            
            print("✅ Task 3 completed successfully!")
            return True
        except Exception as e:
            print(f"❌ Task 3 failed: {str(e)}")
            return False
    
    def generate_final_report(self):
        """Generate comprehensive final report"""
        print(f"\n" + "=" * 60)
        print("STEP 7: FINAL REPORT GENERATION")
        print("=" * 60)
        
        report = self._create_comprehensive_report()
        
        # Save final report
        with open('final_analysis_report.md', 'w', encoding='utf-8') as f:
            f.write(report)
        
        print("📄 Final comprehensive report saved as 'final_analysis_report.md'")
        
        # Generate presentation summary
        presentation_summary = self._create_presentation_summary()
        with open('presentation_summary.md', 'w', encoding='utf-8') as f:
            f.write(presentation_summary)
        
        print("📄 Presentation summary saved as 'presentation_summary.md'")
        
        print("✅ Final report generation completed!")
    
    def _create_comprehensive_report(self) -> str:
        """Create comprehensive analysis report"""
        report = f"""# Comprehensive Superhero & Villain Analysis Report
Group XX - Final Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Executive Summary

This report presents a comprehensive analysis of superhero and villain win probabilities using machine learning techniques. We analyzed {len(self.main_data) if self.main_data is not None else 'N/A'} characters, developed predictive models, and applied them to predict match outcomes and analyze an unbeatable villain.

### Key Findings
"""
        
        # Add key findings from each analysis
        if self.results['eda']:
            report += f"\n#### Exploratory Data Analysis\n"
            if 'role_analysis' in self.results['eda'] and 'hero_villain_comparison' in self.results['eda']['role_analysis']:
                hv_comp = self.results['eda']['role_analysis']['hero_villain_comparison']
                if hv_comp and 'villain_advantage_pct' in hv_comp:
                    advantage = hv_comp['villain_advantage_pct']
                    report += f"- Villains outperform heroes by {advantage:.1f}% on average\n"
            
            if 'high_performers' in self.results['eda']:
                hp = self.results['eda']['high_performers']
                report += f"- {hp['count']} characters ({hp['percentage']:.1f}%) achieve >90% win probability\n"
        
        if self.results['model_performance']:
            best_model = self.results['model_performance']['best_model_name']
            report += f"\n#### Model Performance\n"
            report += f"- Best performing model: {best_model}\n"
            
            if 'test_results' in self.results['model_performance'] and best_model in self.results['model_performance']['test_results']:
                test_metrics = self.results['model_performance']['test_results'][best_model]['metrics']
                report += f"- Test set performance: MAE = {test_metrics['mae']:.4f}, R² = {test_metrics['r2']:.4f}\n"
        
        if self.results['task2']:
            char_predictions = self.results['task2']['character_predictions']
            match_results = self.results['task2']['match_results']
            
            report += f"\n#### Task 2: Match Predictions\n"
            if len(char_predictions) > 0:
                strongest = char_predictions.iloc[0]
                report += f"- Strongest character: {strongest['name']} ({strongest['predicted_win_prob']:.4f})\n"
            
            high_confidence = sum(1 for match in match_results if match['confidence_level'] in ['High', 'Very High'])
            report += f"- High confidence predictions: {high_confidence}/{len(match_results)} matches\n"
        
        if self.results['task3']:
            villain_data = self.task3_analyzer.task3_villain.iloc[0] if self.task3_analyzer else None
            if villain_data is not None:
                report += f"\n#### Task 3: Unbeatable Villain\n"
                report += f"- Analyzed villain: {villain_data.get('name', 'Unknown')}\n"
                
                feature_analysis = self.results['task3']['feature_analysis']
                extreme_features = len([f for f in feature_analysis['extreme_features'] if abs(f['deviation']) > 20])
                report += f"- Extreme advantages identified: {extreme_features}\n"
        
        # Add detailed sections
        report += f"\n## Detailed Analysis\n\n"
        
        # Dataset overview
        if self.main_data is not None:
            report += f"### Dataset Overview\n"
            report += f"- **Total characters**: {len(self.main_data):,}\n"
            report += f"- **Features**: {len(self.main_data.columns)}\n"
            report += f"- **Win probability range**: {self.main_data['win_prob'].min():.3f} - {self.main_data['win_prob'].max():.3f}\n\n"
        
        # Model details
        if self.results['model_performance']:
            report += f"### Model Performance Details\n"
            training_results = self.results['model_performance']['training_results']
            
            report += f"| Model | Validation MAE | Validation R² | CV MAE |\n"
            report += f"|-------|----------------|---------------|--------|\n"
            
            for model_name, results in training_results.items():
                val_mae = results['val_metrics']['mae']
                val_r2 = results['val_metrics']['r2']
                cv_mae = results['cv_mae']
                report += f"| {model_name} | {val_mae:.4f} | {val_r2:.4f} | {cv_mae:.4f} |\n"
            
            report += f"\n"
        
        # Task 2 details
        if self.results['task2']:
            report += f"### Task 2: Detailed Match Predictions\n\n"
            
            # Character rankings
            char_predictions = self.results['task2']['character_predictions']
            report += f"#### Character Win Probabilities\n\n"
            report += f"| Rank | Character | Role | Win Probability |\n"
            report += f"|------|-----------|------|----------------|\n"
            
            for i, (_, char) in enumerate(char_predictions.iterrows()):
                report += f"| {i+1} | {char['name']} | {char['role']} | {char['predicted_win_prob']:.4f} |\n"
            
            # Match outcomes
            report += f"\n#### Match Outcomes\n\n"
            match_results = self.results['task2']['match_results']
            
            for match in match_results:
                report += f"**Match {match['match_number']}**: {match['fighter1']['name']} vs {match['fighter2']['name']}\n"
                report += f"- Winner: {match['winner']} ({match['confidence_level']} confidence)\n"
                report += f"- Win probabilities: {match['fighter1']['prob']:.4f} vs {match['fighter2']['prob']:.4f}\n\n"
        
        # Task 3 details
        if self.results['task3']:
            report += f"### Task 3: Unbeatable Villain Analysis\n\n"
            
            villain_data = self.task3_analyzer.task3_villain.iloc[0]
            report += f"**Villain**: {villain_data.get('name', 'Unknown')}\n\n"
            
            feature_analysis = self.results['task3']['feature_analysis']
            
            # Extreme advantages
            extreme_features = [f for f in feature_analysis['extreme_features'] if abs(f['deviation']) > 20]
            if extreme_features:
                report += f"#### Extreme Advantages\n\n"
                for feature in extreme_features:
                    direction = "higher" if feature['deviation'] > 0 else "lower"
                    report += f"- **{feature['feature']}**: {feature['deviation']:+.1f}% {direction} than average\n"
                report += f"\n"
            
            # Strategic elements
            strategic = feature_analysis['strategic_advantages']
            if strategic:
                report += f"#### Strategic Elements\n\n"
                for element in strategic:
                    report += f"- **{element['feature']}**: {element['value']} ({element['advantage_type']} advantage)\n"
                report += f"\n"
        
        # Conclusions and recommendations
        report += f"## Conclusions and Recommendations\n\n"
        report += f"### Key Insights\n\n"
        report += f"1. **Model Performance**: Successfully developed predictive models with reasonable accuracy\n"
        report += f"2. **Character Analysis**: Identified clear patterns distinguishing high and low performers\n"
        report += f"3. **Match Predictions**: Provided confident predictions for competitive matchups\n"
        report += f"4. **Villain Analysis**: Comprehensively explained what makes certain characters unbeatable\n\n"
        
        report += f"### Methodology Strengths\n\n"
        report += f"- Comprehensive preprocessing pipeline handling missing values and outliers\n"
        report += f"- Multiple model comparison ensuring best performance\n"
        report += f"- Feature importance analysis providing interpretability\n"
        report += f"- Robust evaluation using cross-validation\n\n"
        
        report += f"### Future Improvements\n\n"
        report += f"- Advanced feature engineering (interaction terms, domain-specific features)\n"
        report += f"- Ensemble methods combining multiple algorithms\n"
        report += f"- Deep learning approaches for complex pattern recognition\n"
        report += f"- Expanded dataset for better generalization\n\n"
        
        report += f"---\n"
        report += f"*Report generated by Superhero Analysis Pipeline v1.0*\n"
        
        return report
    
    def _create_presentation_summary(self) -> str:
        """Create presentation summary"""
        summary = f"""# Presentation Summary - Superhero & Villain Analysis
Group XX

## Slide Structure (10 minutes max)

### Slide 1: Title
- **Title**: Superhero and Villain Win Probability Analysis
- **Team**: Group XX
- **Date**: {datetime.now().strftime('%B %Y')}

### Slides 2-3: EDA (1-2 slides)
**Key Findings from Exploratory Data Analysis**

"""
        
        if self.results['eda']:
            if 'role_analysis' in self.results['eda']:
                hv_comp = self.results['eda']['role_analysis'].get('hero_villain_comparison', {})
                if hv_comp:
                    summary += f"- Villains outperform heroes by {hv_comp.get('villain_advantage', 0):.1f}% on average\n"
            
            if 'high_performers' in self.results['eda']:
                hp = self.results['eda']['high_performers']
                summary += f"- {hp['count']} characters achieve >90% win probability\n"
            
            summary += f"- Dataset: {len(self.main_data) if self.main_data is not None else 'N/A'} characters with comprehensive feature analysis\n"
        
        summary += f"""
**Visual**: Use 'eda_visualizations.png'

### Slide 4: Methodology
**Data Processing and Model Selection**

"""
        
        if self.results['preprocessing']:
            summary += f"- Preprocessing: Handled missing values, outliers, and feature encoding\n"
            summary += f"- Features: Selected {len(self.preprocessor.selected_features)} most important features\n"
        
        if self.results['model_performance']:
            best_model = self.results['model_performance']['best_model_name']
            summary += f"- Best Model: {best_model}\n"
            
            if 'test_results' in self.results['model_performance']:
                test_results = self.results['model_performance']['test_results']
                if best_model in test_results:
                    metrics = test_results[best_model]['metrics']
                    summary += f"- Performance: MAE = {metrics['mae']:.4f}, R² = {metrics['r2']:.4f}\n"
        
        summary += f"""
**Visual**: Use 'model_performance.png'

### Slides 5-6: Evaluation (1-2 slides)
**Model Performance and Results**

"""
        
        # Task 2 results
        if self.results['task2']:
            char_predictions = self.results['task2']['character_predictions']
            match_results = self.results['task2']['match_results']
            
            summary += f"**Task 2 - Match Predictions:**\n"
            if len(char_predictions) > 0:
                strongest = char_predictions.iloc[0]
                summary += f"- Strongest: {strongest['name']} ({strongest['predicted_win_prob']:.4f})\n"
            
            summary += f"- Match Winners:\n"
            for match in match_results:
                summary += f"  - Match {match['match_number']}: {match['winner']} ({match['confidence_level']} confidence)\n"
        
        # Task 3 results
        if self.results['task3']:
            villain_data = self.task3_analyzer.task3_villain.iloc[0] if self.task3_analyzer else None
            if villain_data is not None:
                summary += f"\n**Task 3 - Unbeatable Villain:**\n"
                summary += f"- Analyzed: {villain_data.get('name', 'Unknown')}\n"
                
                feature_analysis = self.results['task3']['feature_analysis']
                extreme_count = len([f for f in feature_analysis['extreme_features'] if abs(f['deviation']) > 20])
                summary += f"- Key advantages: {extreme_count} extreme statistical advantages\n"
        
        summary += f"""
**Visuals**: Use 'task2_analysis.png' and 'task3_analysis.png'

### Slide 7: Discussion
**Key Insights and Conclusions**

1. **Successful Model Development**: Achieved reliable win probability predictions
2. **Clear Performance Patterns**: Identified what makes characters successful
3. **Accurate Match Predictions**: Provided confident predictions for all matchups
4. **Comprehensive Villain Analysis**: Explained unbeatable status through feature analysis

**Key Technical Achievements**:
- Robust preprocessing pipeline
- Multiple model comparison and selection
- Feature importance analysis for interpretability
- Comprehensive evaluation methodology

**Business Impact**:
- Can predict fight outcomes with quantified confidence
- Identifies key factors for character strength
- Provides data-driven insights for character development

---

## Talking Points for Each Slide

### EDA Discussion Points
- Highlight the villain advantage over heroes
- Discuss the distribution of win probabilities
- Mention interesting patterns in categorical features

### Methodology Discussion Points
- Emphasize the comprehensive preprocessing approach
- Justify model selection based on performance metrics
- Highlight the importance of cross-validation

### Results Discussion Points
- Walk through Task 2 predictions with confidence levels
- Explain the Task 3 villain's key advantages
- Discuss model accuracy and reliability

### Discussion Points
- Summarize the technical approach and its effectiveness
- Highlight practical applications of the analysis
- Mention potential improvements and future work

## Files to Include in Submission
1. **final_analysis_report.md** - Complete technical report
2. **Group_XX.zip** - Source code archive containing:
   - main.py (this pipeline)
   - data_loader.py
   - eda.py
   - preprocessing.py
   - model_performance.py
   - task2.py
   - task3.py
   - All generated reports and visualizations
3. **Group_XX.pdf** - Presentation slides based on this summary

Good luck with your presentation! 🚀
"""
        
        return summary
    
    def print_final_summary(self):
        """Print final analysis summary"""
        print(f"\n" + "🎉" * 30)
        print("ANALYSIS PIPELINE COMPLETED SUCCESSFULLY!")
        print("🎉" * 30)
        
        print(f"\n📊 FINAL SUMMARY")
        print("=" * 50)
        
        # Data summary
        if self.main_data is not None:
            print(f"📋 Dataset: {len(self.main_data):,} characters analyzed")
        
        # Model summary
        if self.results['model_performance']:
            best_model = self.results['model_performance']['best_model_name']
            print(f"🤖 Best Model: {best_model}")
            
            if 'test_results' in self.results['model_performance']:
                test_results = self.results['model_performance']['test_results']
                if best_model in test_results:
                    metrics = test_results[best_model]['metrics']
                    print(f"📈 Performance: MAE = {metrics['mae']:.4f}, R² = {metrics['r2']:.4f}")
        
        # Task 2 summary
        if self.results['task2']:
            match_results = self.results['task2']['match_results']
            print(f"\n⚔️  Task 2 - Match Predictions:")
            for match in match_results:
                confidence_emoji = "🟢" if match['confidence_level'] in ['High', 'Very High'] else "🟡" if match['confidence_level'] == 'Medium' else "🔴"
                print(f"   {confidence_emoji} Match {match['match_number']}: {match['winner']} wins ({match['confidence_level']})")
        
        # Task 3 summary
        if self.results['task3']:
            villain_data = self.task3_analyzer.task3_villain.iloc[0] if self.task3_analyzer else None
            if villain_data is not None:
                print(f"\n🦹 Task 3 - Unbeatable Villain:")
                print(f"   Analyzed: {villain_data.get('name', 'Unknown')}")
                
                feature_analysis = self.results['task3']['feature_analysis']
                extreme_features = [f for f in feature_analysis['extreme_features'] if abs(f['deviation']) > 20]
                if extreme_features:
                    top_advantage = max(extreme_features, key=lambda x: abs(x['deviation']))
                    print(f"   Key advantage: {top_advantage['feature']} ({top_advantage['deviation']:+.1f}%)")
        
        # Files generated
        print(f"\n📁 Generated Files:")
        files = [
            'eda_visualizations.png',
            'model_performance.png', 
            'task2_analysis.png',
            'task3_analysis.png',
            'final_analysis_report.md',
            'presentation_summary.md'
        ]
        
        for file in files:
            if os.path.exists(file):
                print(f"   ✅ {file}")
            else:
                print(f"   ❌ {file} (not found)")
        
        print(f"\n🎯 Next Steps:")
        print(f"   1. Review 'final_analysis_report.md' for complete technical details")
        print(f"   2. Use 'presentation_summary.md' to prepare your 10-minute presentation")
        print(f"   3. Include all visualization PNG files in your slides")
        print(f"   4. Submit the code as 'Group_XX.zip' and presentation as 'Group_XX.pdf'")
        
        print(f"\n🚀 Ready for submission and presentation!")
    
    def run_individual_task(self, task: str) -> bool:
        """
        Run individual task for debugging/testing
        
        Args:
            task (str): Task to run ('eda', 'preprocessing', 'model', 'task2', 'task3')
            
        Returns:
            bool: Success status
        """
        print(f"🎯 Running individual task: {task.upper()}")
        
        # Ensure data is loaded
        if self.main_data is None:
            if not self.load_data():
                return False
        
        if task.lower() == 'eda':
            return self.run_eda()
        
        elif task.lower() == 'preprocessing':
            return self.run_preprocessing()
        
        elif task.lower() == 'model':
            if not hasattr(self, 'X_train'):
                print("❌ Preprocessing required before model training")
                if not self.run_preprocessing():
                    return False
            return self.run_model_training()
        
        elif task.lower() == 'task2':
            if not hasattr(self, 'X_train') or self.model_analyzer.best_model is None:
                print("❌ Model training required before Task 2")
                if not self.run_preprocessing() or not self.run_model_training():
                    return False
            return self.run_task2()
        
        elif task.lower() == 'task3':
            if not hasattr(self, 'X_train') or self.model_analyzer.best_model is None:
                print("❌ Model training required before Task 3")
                if not self.run_preprocessing() or not self.run_model_training():
                    return False
            return self.run_task3()
        
        else:
            print(f"❌ Unknown task: {task}")
            return False
    
    def run_complete_pipeline(self) -> bool:
        """
        Run the complete analysis pipeline
        
        Returns:
            bool: True if pipeline completed successfully
        """
        try:
            # Setup
            self.setup_environment()
            
            # Step 1: Load data
            if not self.load_data():
                return False
            
            # Step 2: EDA
            if not self.run_eda():
                return False
            
            # Step 3: Preprocessing
            if not self.run_preprocessing():
                return False
            
            # Step 4: Model training
            if not self.run_model_training():
                return False
            
            # Step 5: Task 2
            if self.config['run_all_tasks']:
                if not self.run_task2():
                    return False
            
            # Step 6: Task 3
            if self.config['run_all_tasks']:
                if not self.run_task3():
                    return False
            
            # Step 7: Final report
            if self.config['generate_reports']:
                self.generate_final_report()
            
            # Summary
            self.print_final_summary()
            
            return True
            
        except Exception as e:
            print(f"\n❌ Pipeline failed with error: {str(e)}")
            import traceback
            traceback.print_exc()
            return False

def main():
    """Main function with command line interface"""
    parser = argparse.ArgumentParser(
        description="Superhero & Villain Win Probability Analysis Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                    # Run complete pipeline
  python main.py --task eda         # Run only EDA
  python main.py --task task2       # Run only Task 2 (requires model)
  python main.py --no-viz          # Skip visualizations
  python main.py --no-reports      # Skip report generation
        """
    )
    
    parser.add_argument(
        '--task', 
        type=str, 
        choices=['eda', 'preprocessing', 'model', 'task2', 'task3'],
        help='Run specific task only'
    )
    
    parser.add_argument(
        '--no-viz', 
        action='store_true',
        help='Skip visualization generation'
    )
    
    parser.add_argument(
        '--no-reports', 
        action='store_true',
        help='Skip report generation'
    )
    
    parser.add_argument(
        '--no-save', 
        action='store_true',
        help='Skip saving models and pipelines'
    )
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = SuperheroAnalysisPipeline()
    
    # Configure based on arguments
    pipeline.config['create_visualizations'] = not args.no_viz
    pipeline.config['generate_reports'] = not args.no_reports
    pipeline.config['save_results'] = not args.no_save
    
    # Run pipeline
    if args.task:
        # Run individual task
        success = pipeline.run_individual_task(args.task)
    else:
        # Run complete pipeline
        success = pipeline.run_complete_pipeline()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()

# Alternative usage for Jupyter notebooks or direct Python execution
def run_analysis():
    """
    Convenience function for running analysis in Jupyter notebooks
    """
    pipeline = SuperheroAnalysisPipeline()
    return pipeline.run_complete_pipeline()

def run_task(task_name: str):
    """
    Convenience function for running individual tasks
    
    Args:
        task_name (str): Name of task to run
    """
    pipeline = SuperheroAnalysisPipeline()
    return pipeline.run_individual_task(task_name)

# Example usage patterns:
"""
# Complete analysis
if __name__ == "__main__":
    success = run_analysis()
    print(f"Analysis {'completed' if success else 'failed'}")

# Individual tasks
if __name__ == "__main__":
    # Run only EDA
    run_task('eda')
    
    # Run only Task 2
    run_task('task2')
    
    # Run only Task 3
    run_task('task3')

# Custom pipeline
if __name__ == "__main__":
    pipeline = SuperheroAnalysisPipeline()
    
    # Load data
    pipeline.load_data()
    
    # Run specific components
    pipeline.run_eda()
    pipeline.run_preprocessing()
    pipeline.run_model_training()
    
    # Generate custom report
    pipeline.generate_final_report()
"""