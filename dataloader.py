"""
Data Loader Module
Handles loading and basic validation of all CSV files for the superhero analysis project.
"""

import pandas as pd
import os
from typing import Tuple, Optional

class DataLoader:
    """
    Class for loading and validating superhero/villain datasets
    """
    
    def __init__(self, data_dir: str = "."):
        """
        Initialize DataLoader
        
        Args:
            data_dir (str): Directory containing the CSV files
        """
        self.data_dir = data_dir
        self.main_data = None
        self.task2_characters = None
        self.task2_matches = None
        self.task3_villain = None
        
        # Expected file names
        self.file_names = {
            'main': 'data.csv',
            'task2_chars': 'Task2_superheroes_villains.csv',
            'task2_matches': 'Task2_matches.csv',
            'task3_villain': 'Task3_villain.csv'
        }
    
    def validate_files(self) -> bool:
        """
        Check if all required files exist
        
        Returns:
            bool: True if all files exist, False otherwise
        """
        missing_files = []
        
        for file_type, filename in self.file_names.items():
            filepath = os.path.join(self.data_dir, filename)
            if not os.path.exists(filepath):
                missing_files.append(filename)
        
        if missing_files:
            print("  Missing files:")
            for file in missing_files:
                print(f"   - {file}")
            return False
        
        print("  All required files found!")
        return True
    
    def load_main_data(self) -> pd.DataFrame:
        """
        Load main dataset
        
        Returns:
            pd.DataFrame: Main dataset
        """
        try:
            filepath = os.path.join(self.data_dir, self.file_names['main'])
            self.main_data = pd.read_csv(filepath)
            print(f"  Main dataset loaded: {self.main_data.shape[0]} rows, {self.main_data.shape[1]} columns")
            return self.main_data
        except Exception as e:
            print(f"  Error loading main dataset: {e}")
            return None
    
    def load_task2_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load Task 2 datasets
        
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: Characters and matches dataframes
        """
        try:
            chars_path = os.path.join(self.data_dir, self.file_names['task2_chars'])
            matches_path = os.path.join(self.data_dir, self.file_names['task2_matches'])
            
            self.task2_characters = pd.read_csv(chars_path)
            self.task2_matches = pd.read_csv(matches_path)
            
            print(f"  Task 2 characters loaded: {self.task2_characters.shape[0]} characters")
            print(f"  Task 2 matches loaded: {self.task2_matches.shape[0]} matches")
            
            return self.task2_characters, self.task2_matches
        except Exception as e:
            print(f"  Error loading Task 2 data: {e}")
            return None, None
    
    def load_task3_data(self) -> pd.DataFrame:
        """
        Load Task 3 dataset
        
        Returns:
            pd.DataFrame: Task 3 villain data
        """
        try:
            filepath = os.path.join(self.data_dir, self.file_names['task3_villain'])
            self.task3_villain = pd.read_csv(filepath)
            print(f"  Task 3 villain loaded: {self.task3_villain.shape[0]} villain(s)")
            return self.task3_villain
        except Exception as e:
            print(f"  Error loading Task 3 data: {e}")
            return None
    
    def load_all_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Load all datasets at once
        
        Returns:
            Tuple: (main_data, task2_characters, task2_matches, task3_villain)
        """
        print(" Loading all datasets...")
        print("=" * 50)
        
        if not self.validate_files():
            return None, None, None, None
        
        # Load all datasets
        main_data = self.load_main_data()
        task2_chars, task2_matches = self.load_task2_data()
        task3_villain = self.load_task3_data()
        
        if all(data is not None for data in [main_data, task2_chars, task2_matches, task3_villain]):
            print("\n  All datasets loaded successfully!")
            self._print_summary()
            return main_data, task2_chars, task2_matches, task3_villain
        else:
            print("\n  Failed to load some datasets!")
            return None, None, None, None
    
    def _print_summary(self):
        """Print summary of loaded datasets"""
        print("\n Dataset Summary:")
        print("-" * 30)
        
        if self.main_data is not None:
            print(f"Main Dataset: {self.main_data.shape[0]} rows × {self.main_data.shape[1]} columns")
            print(f"  - Target variable range: {self.main_data['win_prob'].min():.3f} to {self.main_data['win_prob'].max():.3f}")
            print(f"  - Memory usage: {self.main_data.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        if self.task2_characters is not None:
            print(f"Task 2 Characters: {self.task2_characters.shape[0]} characters")
            roles = self.task2_characters['role'].value_counts()
            print(f"  - Heroes: {roles.get('Hero', 0) + roles.get(' Hero', 0)}")
            print(f"  - Villains: {roles.get('Villain', 0)}")
        
        if self.task2_matches is not None:
            print(f"Task 2 Matches: {self.task2_matches.shape[0]} matches")
        
        if self.task3_villain is not None:
            villain_name = self.task3_villain.iloc[0]['name'] if len(self.task3_villain) > 0 else "Unknown"
            print(f"Task 3 Villain: {villain_name}")
    
    def get_feature_info(self) -> dict:
        """
        Get information about features in the main dataset
        
        Returns:
            dict: Feature information
        """
        if self.main_data is None:
            return {}
        
        numerical_features = [
            'power_level', 'weight', 'height', 'age', 'gender', 'speed', 
            'battle_iq', 'ranking', 'intelligence', 'training_time', 'secret_code'
        ]
        
        categorical_features = [
            'role', 'skin_type', 'eye_color', 'hair_color', 'universe', 
            'body_type', 'job', 'species', 'abilities', 'special_attack'
        ]
        
        # Filter features that actually exist in the dataset
        existing_numerical = [f for f in numerical_features if f in self.main_data.columns]
        existing_categorical = [f for f in categorical_features if f in self.main_data.columns]
        
        return {
            'numerical_features': existing_numerical,
            'categorical_features': existing_categorical,
            'target_feature': 'win_prob',
            'total_features': len(self.main_data.columns),
            'feature_types': {
                'numerical': len(existing_numerical),
                'categorical': len(existing_categorical),
                'other': len(self.main_data.columns) - len(existing_numerical) - len(existing_categorical) - 1  # -1 for target
            }
        }

# Example usage and testing
if __name__ == "__main__":
    # Test the DataLoader
    loader = DataLoader()
    
    # Load all data
    main_data, task2_chars, task2_matches, task3_villain = loader.load_all_data()
    
    if main_data is not None:
        # Show feature information
        feature_info = loader.get_feature_info()
        print(f"\n Feature Information:")
        print(f"  - Numerical features: {feature_info['feature_types']['numerical']}")
        print(f"  - Categorical features: {feature_info['feature_types']['categorical']}")
        print(f"  - Total features: {feature_info['total_features']}")
        
        # Show first few rows
        print(f"\n Sample Data (first 3 rows):")
        print(main_data.head(3).to_string())
    else:
        print("Failed to load data. Please check if all CSV files are present.")