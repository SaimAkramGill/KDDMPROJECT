"""
Task 2 Module: Match Predictions
Handles prediction of win probabilities for Task 2 characters and match outcomes
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

class Task2MatchPredictor:
    """
    Class for handling Task 2 match predictions and analysis
    """
    
    def __init__(self, model, preprocessor):
        """
        Initialize Task 2 predictor
        
        Args:
            model: Trained model for predictions
            preprocessor: Fitted data preprocessor
        """
        self.model = model
        self.preprocessor = preprocessor
        self.task2_characters = None
        self.task2_matches = None
        self.character_predictions = None
        self.match_results = None
        
    def load_task2_data(self, characters_data: pd.DataFrame, matches_data: pd.DataFrame):
        """
        Load Task 2 data
        
        Args:
            characters_data (pd.DataFrame): Task 2 characters data
            matches_data (pd.DataFrame): Task 2 matches data
        """
        self.task2_characters = characters_data.copy()
        self.task2_matches = matches_data.copy()
        
        print("=" * 60)
        print("TASK 2: MATCH PREDICTIONS")
        print("=" * 60)
        
        print(f"📊 Task 2 Data Loaded:")
        print(f"   Characters: {len(self.task2_characters)}")
        print(f"   Matches: {len(self.task2_matches)}")
        
        # Display character information
        print(f"\n🦸 Characters in Task 2:")
        for _, char in self.task2_characters.iterrows():
            role = char.get('role', 'Unknown').strip()
            name = char.get('name', 'Unknown')
            print(f"   - {name} ({role})")
        
        # Display matches
        print(f"\n⚔️  Matches to predict:")
        for i, match in self.task2_matches.iterrows():
            print(f"   Match {i+1}: {match['first']} vs {match['second']}")
    
    def preprocess_task2_characters(self) -> pd.DataFrame:
        """
        Preprocess Task 2 characters using the fitted preprocessor
        
        Returns:
            pd.DataFrame: Preprocessed character data
        """
        print(f"\n🔄 Preprocessing Task 2 characters...")
        
        if self.task2_characters is None:
            raise ValueError("Task 2 characters data not loaded")
        
        # Apply preprocessing pipeline
        processed_characters = self.preprocessor.transform_new_data(self.task2_characters)
        
        print(f"✅ Task 2 characters preprocessed successfully")
        print(f"   Original features: {len(self.task2_characters.columns)}")
        print(f"   Processed features: {len(processed_characters.columns)}")
        
        return processed_characters
    
    def predict_character_win_probabilities(self) -> pd.DataFrame:
        """
        Predict win probabilities for all Task 2 characters
        
        Returns:
            pd.DataFrame: Character predictions
        """
        print(f"\n🎯 Predicting win probabilities for Task 2 characters...")
        
        # Preprocess characters
        processed_characters = self.preprocess_task2_characters()
        
        # Extract features for prediction
        X_task2 = processed_characters[self.preprocessor.selected_features]
        
        # Make predictions
        predictions = self.model.predict(X_task2)
        
        # Create results dataframe
        results = pd.DataFrame({
            'name': self.task2_characters['name'],
            'role': self.task2_characters['role'],
            'predicted_win_prob': predictions
        })
        
        # Sort by predicted win probability
        results = results.sort_values('predicted_win_prob', ascending=False)
        
        print(f"\n📊 Character Win Probability Predictions:")
        print("-" * 50)
        for i, (_, char) in enumerate(results.iterrows()):
            rank_emoji = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else "  "
            role_emoji = "🦹" if 'villain' in char['role'].lower() else "🦸"
            print(f"{rank_emoji} {i+1}. {char['name']} ({char['role']}) {role_emoji}: {char['predicted_win_prob']:.4f}")
        
        self.character_predictions = results
        return results
    
    def predict_match_outcomes(self) -> List[Dict]:
        """
        Predict outcomes for all Task 2 matches
        
        Returns:
            List[Dict]: Match prediction results
        """
        print(f"\n⚔️  Predicting match outcomes...")
        
        if self.character_predictions is None:
            self.predict_character_win_probabilities()
        
        match_results = []
        
        print(f"\n📋 Match Predictions:")
        print("=" * 60)
        
        for i, match in self.task2_matches.iterrows():
            fighter1_name = match['first']
            fighter2_name = match['second']
            
            # Get predictions for both fighters
            fighter1_data = self.character_predictions[
                self.character_predictions['name'] == fighter1_name
            ]
            fighter2_data = self.character_predictions[
                self.character_predictions['name'] == fighter2_name
            ]
            
            if fighter1_data.empty or fighter2_data.empty:
                print(f"❌ Could not find prediction data for match: {fighter1_name} vs {fighter2_name}")
                continue
            
            fighter1_prob = fighter1_data['predicted_win_prob'].iloc[0]
            fighter2_prob = fighter2_data['predicted_win_prob'].iloc[0]
            fighter1_role = fighter1_data['role'].iloc[0]
            fighter2_role = fighter2_data['role'].iloc[0]
            
            # Determine winner
            winner = fighter1_name if fighter1_prob > fighter2_prob else fighter2_name
            loser = fighter2_name if fighter1_prob > fighter2_prob else fighter1_name
            winner_prob = max(fighter1_prob, fighter2_prob)
            loser_prob = min(fighter1_prob, fighter2_prob)
            
            # Calculate confidence metrics
            prob_difference = abs(fighter1_prob - fighter2_prob)
            relative_advantage = (prob_difference / ((fighter1_prob + fighter2_prob) / 2)) * 100
            
            # Categorize match closeness
            if relative_advantage < 5:
                closeness = "Very Close"
                confidence_level = "Low"
            elif relative_advantage < 15:
                closeness = "Close"
                confidence_level = "Medium"
            elif relative_advantage < 30:
                closeness = "Clear"
                confidence_level = "High"
            else:
                closeness = "Dominant"
                confidence_level = "Very High"
            
            match_result = {
                'match_number': i + 1,
                'fighter1': {
                    'name': fighter1_name,
                    'role': fighter1_role,
                    'prob': fighter1_prob
                },
                'fighter2': {
                    'name': fighter2_name,
                    'role': fighter2_role,
                    'prob': fighter2_prob
                },
                'winner': winner,
                'loser': loser,
                'winner_prob': winner_prob,
                'loser_prob': loser_prob,
                'prob_difference': prob_difference,
                'relative_advantage': relative_advantage,
                'closeness': closeness,
                'confidence_level': confidence_level
            }
            
            match_results.append(match_result)
            
            # Display match prediction
            print(f"\n🥊 Match {i+1}: {fighter1_name} vs {fighter2_name}")
            print(f"   {fighter1_name} ({fighter1_role}): {fighter1_prob:.4f}")
            print(f"   {fighter2_name} ({fighter2_role}): {fighter2_prob:.4f}")
            print(f"   ")
            print(f"   🏆 Predicted Winner: {winner}")
            print(f"   📊 Confidence: {confidence_level} ({relative_advantage:.2f}% advantage)")
            print(f"   🎯 Match Type: {closeness}")
            
            # Add special notes for interesting matches
            if relative_advantage < 5:
                print(f"   ⚠️  This is an extremely close match - could go either way!")
            elif relative_advantage > 50:
                print(f"   💪 This looks like a dominant performance!")
            
            if 'hero' in fighter1_role.lower() and 'villain' in fighter2_role.lower():
                hero = fighter1_name if 'hero' in fighter1_role.lower() else fighter2_name
                villain = fighter2_name if 'hero' in fighter1_role.lower() else fighter1_name
                winner_type = "Hero" if 'hero' in self.character_predictions[
                    self.character_predictions['name'] == winner]['role'].iloc[0].lower() else "Villain"
                print(f"   ⚡ Classic Hero vs Villain matchup - {winner_type} predicted to win!")
        
        self.match_results = match_results
        return match_results
    
    def analyze_prediction_patterns(self) -> Dict:
        """
        Analyze patterns in the predictions
        
        Returns:
            Dict: Analysis results
        """
        print(f"\n📈 Analyzing prediction patterns...")
        
        if self.character_predictions is None or self.match_results is None:
            raise ValueError("Predictions not available. Run predictions first.")
        
        analysis = {}
        
        # 1. Role performance analysis
        role_analysis = self.character_predictions.groupby('role').agg({
            'predicted_win_prob': ['mean', 'min', 'max', 'count']
        }).round(4)
        
        role_analysis.columns = ['Mean_Prob', 'Min_Prob', 'Max_Prob', 'Count']
        analysis['role_performance'] = role_analysis
        
        print(f"\n📊 Role Performance Analysis:")
        print(role_analysis.to_string())
        
        # 2. Hero vs Villain analysis
        hero_characters = self.character_predictions[
            self.character_predictions['role'].str.contains('hero', case=False, na=False)
        ]
        villain_characters = self.character_predictions[
            self.character_predictions['role'].str.contains('villain', case=False, na=False)
        ]
        
        if len(hero_characters) > 0 and len(villain_characters) > 0:
            hero_avg = hero_characters['predicted_win_prob'].mean()
            villain_avg = villain_characters['predicted_win_prob'].mean()
            
            analysis['hero_vs_villain'] = {
                'hero_average': hero_avg,
                'villain_average': villain_avg,
                'villain_advantage': ((villain_avg - hero_avg) / hero_avg) * 100 if hero_avg > 0 else 0,
                'hero_count': len(hero_characters),
                'villain_count': len(villain_characters)
            }
            
            print(f"\n🦸 Hero vs Villain Analysis:")
            print(f"   Heroes: {len(hero_characters)} characters, avg prob: {hero_avg:.4f}")
            print(f"   Villains: {len(villain_characters)} characters, avg prob: {villain_avg:.4f}")
            
            if villain_avg > hero_avg:
                advantage = ((villain_avg - hero_avg) / hero_avg) * 100
                print(f"   🦹 Villains have {advantage:.1f}% higher average win probability")
            else:
                advantage = ((hero_avg - villain_avg) / villain_avg) * 100
                print(f"   🦸 Heroes have {advantage:.1f}% higher average win probability")
        
        # 3. Match closeness analysis
        if self.match_results:
            closeness_counts = {}
            confidence_counts = {}
            
            for match in self.match_results:
                closeness = match['closeness']
                confidence = match['confidence_level']
                
                closeness_counts[closeness] = closeness_counts.get(closeness, 0) + 1
                confidence_counts[confidence] = confidence_counts.get(confidence, 0) + 1
            
            analysis['match_analysis'] = {
                'closeness_distribution': closeness_counts,
                'confidence_distribution': confidence_counts,
                'total_matches': len(self.match_results)
            }
            
            print(f"\n⚔️  Match Analysis:")
            print(f"   Match Closeness Distribution:")
            for closeness, count in closeness_counts.items():
                print(f"     {closeness}: {count} matches")
            
            print(f"   Confidence Level Distribution:")
            for confidence, count in confidence_counts.items():
                print(f"     {confidence}: {count} matches")
        
        # 4. Prediction range analysis
        min_prob = self.character_predictions['predicted_win_prob'].min()
        max_prob = self.character_predictions['predicted_win_prob'].max()
        mean_prob = self.character_predictions['predicted_win_prob'].mean()
        std_prob = self.character_predictions['predicted_win_prob'].std()
        
        analysis['prediction_stats'] = {
            'min_probability': min_prob,
            'max_probability': max_prob,
            'mean_probability': mean_prob,
            'std_probability': std_prob,
            'range': max_prob - min_prob
        }
        
        print(f"\n📊 Prediction Statistics:")
        print(f"   Range: {min_prob:.4f} to {max_prob:.4f}")
        print(f"   Mean: {mean_prob:.4f} ± {std_prob:.4f}")
        print(f"   Spread: {max_prob - min_prob:.4f}")
        
        return analysis
    
    def create_task2_visualizations(self, save_path: str = 'task2_analysis.png'):
        """
        Create comprehensive visualizations for Task 2
        
        Args:
            save_path (str): Path to save visualizations
        """
        print(f"\n🎨 Creating Task 2 visualizations...")
        
        if self.character_predictions is None or self.match_results is None:
            raise ValueError("Predictions not available. Run predictions first.")
        
        # Set up the figure
        fig = plt.figure(figsize=(20, 12))
        
        # 1. Character win probabilities
        plt.subplot(2, 3, 1)
        characters = self.character_predictions.copy()
        colors = ['red' if 'villain' in role.lower() else 'blue' if 'hero' in role.lower() else 'gray' 
                 for role in characters['role']]
        
        bars = plt.bar(range(len(characters)), characters['predicted_win_prob'], color=colors)
        plt.title('Character Win Probabilities', fontweight='bold', fontsize=14)
        plt.xlabel('Characters')
        plt.ylabel('Predicted Win Probability')
        plt.xticks(range(len(characters)), characters['name'], rotation=45, ha='right')
        
        # Add value labels on bars
        for i, bar in enumerate(bars):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=10)
        
        # Add legend
        hero_patch = plt.Rectangle((0, 0), 1, 1, facecolor='blue', label='Heroes')
        villain_patch = plt.Rectangle((0, 0), 1, 1, facecolor='red', label='Villains')
        plt.legend(handles=[hero_patch, villain_patch], loc='upper right')
        
        # 2. Match predictions overview
        plt.subplot(2, 3, 2)
        match_names = [f"Match {match['match_number']}" for match in self.match_results]
        advantages = [match['relative_advantage'] for match in self.match_results]
        
        colors_confidence = []
        for match in self.match_results:
            if match['confidence_level'] == 'Very High':
                colors_confidence.append('darkgreen')
            elif match['confidence_level'] == 'High':
                colors_confidence.append('green')
            elif match['confidence_level'] == 'Medium':
                colors_confidence.append('orange')
            else:
                colors_confidence.append('red')
        
        bars = plt.bar(match_names, advantages, color=colors_confidence)
        plt.title('Match Prediction Confidence', fontweight='bold', fontsize=14)
        plt.xlabel('Matches')
        plt.ylabel('Winner Advantage (%)')
        
        # Add value labels
        for i, bar in enumerate(bars):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{height:.1f}%', ha='center', va='bottom', fontsize=10)
        
        # 3. Head-to-head comparisons
        plt.subplot(2, 3, 3)
        match_positions = []
        probabilities = []
        fighter_names = []
        winner_indicators = []
        
        for i, match in enumerate(self.match_results):
            # Fighter 1
            match_positions.append(i * 3)
            probabilities.append(match['fighter1']['prob'])
            fighter_names.append(match['fighter1']['name'])
            winner_indicators.append('gold' if match['fighter1']['name'] == match['winner'] else 'lightblue')
            
            # Fighter 2
            match_positions.append(i * 3 + 1)
            probabilities.append(match['fighter2']['prob'])
            fighter_names.append(match['fighter2']['name'])
            winner_indicators.append('gold' if match['fighter2']['name'] == match['winner'] else 'lightblue')
        
        bars = plt.bar(match_positions, probabilities, color=winner_indicators, width=0.8)
        plt.title('Head-to-Head Comparisons', fontweight='bold', fontsize=14)
        plt.xlabel('Matches')
        plt.ylabel('Win Probability')
        
        # Set x-axis labels
        match_centers = [i * 3 + 0.5 for i in range(len(self.match_results))]
        plt.xticks(match_centers, [f"Match {i+1}" for i in range(len(self.match_results))])
        
        # Add fighter name labels
        for i, (pos, name, prob) in enumerate(zip(match_positions, fighter_names, probabilities)):
            plt.text(pos, -0.1, name, rotation=45, ha='right', va='top', fontsize=8)
            plt.text(pos, prob + 0.02, f'{prob:.3f}', ha='center', va='bottom', fontsize=9)
        
        # 4. Role performance comparison
        plt.subplot(2, 3, 4)
        role_stats = self.character_predictions.groupby('role')['predicted_win_prob'].agg(['mean', 'count'])
        
        role_colors = ['red' if 'villain' in role.lower() else 'blue' if 'hero' in role.lower() else 'gray' 
                      for role in role_stats.index]
        
        bars = plt.bar(range(len(role_stats)), role_stats['mean'], color=role_colors)
        plt.title('Average Win Probability by Role', fontweight='bold', fontsize=14)
        plt.xlabel('Role')
        plt.ylabel('Average Win Probability')
        plt.xticks(range(len(role_stats)), role_stats.index, rotation=45, ha='right')
        
        # Add count labels
        for i, (bar, count) in enumerate(zip(bars, role_stats['count'])):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.3f}\n(n={count})', ha='center', va='bottom', fontsize=9)
        
        # 5. Match closeness distribution
        plt.subplot(2, 3, 5)
        closeness_counts = {}
        for match in self.match_results:
            closeness = match['closeness']
            closeness_counts[closeness] = closeness_counts.get(closeness, 0) + 1
        
        closeness_order = ['Very Close', 'Close', 'Clear', 'Dominant']
        closeness_values = [closeness_counts.get(c, 0) for c in closeness_order]
        closeness_colors = ['red', 'orange', 'yellow', 'green']
        
        plt.pie(closeness_values, labels=closeness_order, colors=closeness_colors, autopct='%1.0f')
        plt.title('Match Closeness Distribution', fontweight='bold', fontsize=14)
        
        # 6. Prediction uncertainty analysis
        plt.subplot(2, 3, 6)
        uncertainties = [match['prob_difference'] for match in self.match_results]
        match_labels = [f"Match {match['match_number']}" for match in self.match_results]
        
        bars = plt.bar(match_labels, uncertainties, color='purple', alpha=0.7)
        plt.title('Prediction Uncertainty\n(Probability Difference)', fontweight='bold', fontsize=14)
        plt.xlabel('Matches')
        plt.ylabel('Probability Difference')
        
        # Add value labels
        for i, bar in enumerate(bars):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"✅ Task 2 visualizations saved as '{save_path}'")
    
    def generate_task2_report(self) -> str:
        """
        Generate comprehensive Task 2 report
        
        Returns:
            str: Task 2 report
        """
        if self.character_predictions is None or self.match_results is None:
            raise ValueError("Predictions not available. Run predictions first.")
        
        report = "# Task 2: Match Predictions Report\n\n"
        
        # Character predictions
        report += "## Character Win Probability Predictions\n\n"
        report += "| Rank | Character | Role | Predicted Win Probability |\n"
        report += "|------|-----------|------|---------------------------|\n"
        
        for i, (_, char) in enumerate(self.character_predictions.iterrows()):
            report += f"| {i+1} | {char['name']} | {char['role']} | {char['predicted_win_prob']:.4f} |\n"
        
        # Match predictions
        report += "\n## Match Outcome Predictions\n\n"
        
        for match in self.match_results:
            report += f"### Match {match['match_number']}: {match['fighter1']['name']} vs {match['fighter2']['name']}\n\n"
            report += f"**Fighters:**\n"
            report += f"- {match['fighter1']['name']} ({match['fighter1']['role']}): {match['fighter1']['prob']:.4f}\n"
            report += f"- {match['fighter2']['name']} ({match['fighter2']['role']}): {match['fighter2']['prob']:.4f}\n\n"
            report += f"**Prediction:**\n"
            report += f"- **Winner:** {match['winner']}\n"
            report += f"- **Confidence:** {match['confidence_level']} ({match['relative_advantage']:.2f}% advantage)\n"
            report += f"- **Match Type:** {match['closeness']}\n\n"
            
            if match['relative_advantage'] < 5:
                report += "⚠️ **Note:** This is an extremely close match with high uncertainty.\n\n"
            elif match['relative_advantage'] > 50:
                report += "💪 **Note:** This prediction shows a dominant performance.\n\n"
        
        # Summary statistics
        report += "## Summary Statistics\n\n"
        
        # Role performance
        role_stats = self.character_predictions.groupby('role')['predicted_win_prob'].agg(['mean', 'count'])
        report += "### Role Performance\n\n"
        report += "| Role | Count | Average Win Probability |\n"
        report += "|------|-------|------------------------|\n"
        
        for role, stats in role_stats.iterrows():
            report += f"| {role} | {stats['count']} | {stats['mean']:.4f} |\n"
        
        # Match analysis
        report += "\n### Match Analysis\n\n"
        
        closeness_counts = {}
        for match in self.match_results:
            closeness = match['closeness']
            closeness_counts[closeness] = closeness_counts.get(closeness, 0) + 1
        
        report += "**Match Closeness Distribution:**\n"
        for closeness, count in closeness_counts.items():
            report += f"- {closeness}: {count} match(es)\n"
        
        # Key insights
        report += "\n## Key Insights\n\n"
        
        # Highest and lowest performers
        best_char = self.character_predictions.iloc[0]
        worst_char = self.character_predictions.iloc[-1]
        
        report += f"1. **Strongest Character:** {best_char['name']} ({best_char['role']}) with {best_char['predicted_win_prob']:.4f} win probability\n"
        report += f"2. **Weakest Character:** {worst_char['name']} ({worst_char['role']}) with {worst_char['predicted_win_prob']:.4f} win probability\n"
        
        # Hero vs Villain analysis
        hero_chars = self.character_predictions[
            self.character_predictions['role'].str.contains('hero', case=False, na=False)
        ]
        villain_chars = self.character_predictions[
            self.character_predictions['role'].str.contains('villain', case=False, na=False)
        ]
        
        if len(hero_chars) > 0 and len(villain_chars) > 0:
            hero_avg = hero_chars['predicted_win_prob'].mean()
            villain_avg = villain_chars['predicted_win_prob'].mean()
            
            if villain_avg > hero_avg:
                advantage = ((villain_avg - hero_avg) / hero_avg) * 100
                report += f"3. **Villains outperform Heroes** by {advantage:.1f}% on average\n"
            else:
                advantage = ((hero_avg - villain_avg) / villain_avg) * 100
                report += f"3. **Heroes outperform Villains** by {advantage:.1f}% on average\n"
        
        # Match confidence
        high_confidence = sum(1 for match in self.match_results if match['confidence_level'] in ['High', 'Very High'])
        total_matches = len(self.match_results)
        
        report += f"4. **Prediction Confidence:** {high_confidence}/{total_matches} matches have high confidence predictions\n"
        
        return report
    
    def run_complete_task2_analysis(self) -> Dict:
        """
        Run complete Task 2 analysis pipeline
        
        Returns:
            Dict: Complete Task 2 results
        """
        print("🎯 Starting Complete Task 2 Analysis")
        print("=" * 80)
        
        # Step 1: Predict character win probabilities
        character_predictions = self.predict_character_win_probabilities()
        
        # Step 2: Predict match outcomes
        match_results = self.predict_match_outcomes()
        
        # Step 3: Analyze patterns
        analysis = self.analyze_prediction_patterns()
        
        # Step 4: Create visualizations
        self.create_task2_visualizations()
        
        # Step 5: Generate report
        report = self.generate_task2_report()
        with open('task2_report.md', 'w', encoding='utf-8') as f:
            f.write(report)
        print("📄 Task 2 report saved as 'task2_report.md'")
        
        print("\n" + "🎉" * 20)
        print("TASK 2 ANALYSIS COMPLETED SUCCESSFULLY!")
        print("🎉" * 20)
        
        results = {
            'character_predictions': character_predictions,
            'match_results': match_results,
            'analysis': analysis,
            'report': report
        }
        
        # Print summary
        print(f"\n📊 Task 2 Summary:")
        print(f"   Characters analyzed: {len(character_predictions)}")
        print(f"   Matches predicted: {len(match_results)}")
        print(f"   Strongest character: {character_predictions.iloc[0]['name']} ({character_predictions.iloc[0]['predicted_win_prob']:.4f})")
        
        # Print match winners
        print(f"\n🏆 Match Winners:")
        for match in match_results:
            print(f"   Match {match['match_number']}: {match['winner']} ({match['confidence_level']} confidence)")
        
        return results

# Example usage
if __name__ == "__main__":
    from dataloader import DataLoader
    from preprocessing import DataPreprocessor
    from model_performance import ModelPerformanceAnalyzer
    
    # Load data
    loader = DataLoader()
    main_data, task2_chars, task2_matches, _ = loader.load_all_data()
    
    if all(data is not None for data in [main_data, task2_chars, task2_matches]):
        # Preprocess data and train model
        preprocessor = DataPreprocessor()
        X_train, X_val, X_test, y_train, y_val, y_test, _ = preprocessor.run_complete_preprocessing(main_data)
        
        # Train model
        analyzer = ModelPerformanceAnalyzer()
        analyzer.train_models(X_train, y_train, X_val, y_val)
        
        # Run Task 2 analysis
        task2_predictor = Task2MatchPredictor(analyzer.best_model, preprocessor)
        task2_predictor.load_task2_data(task2_chars, task2_matches)
        results = task2_predictor.run_complete_task2_analysis()
        
        print(f"\n🎯 Task 2 Complete!")
    else:
        print("❌ Failed to load required data for Task 2 analysis")