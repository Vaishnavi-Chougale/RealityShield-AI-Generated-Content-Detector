#!/usr/bin/env python3
"""
Simple demo runner for the Multidisciplinary Deepfake Detection project
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def check_project_structure():
    """Check if required directories exist"""
    required_dirs = ['data', 'models', 'logs', 'src']
    missing_dirs = []
    
    for dir_name in required_dirs:
        if not os.path.exists(dir_name):
            missing_dirs.append(dir_name)
    
    if missing_dirs:
        print(f"Missing directories: {missing_dirs}")
        return False
    return True

def create_sample_data():
    """Create sample data for demonstration"""
    print("Creating sample data...")
    
    # Create sample CSV data
    sample_data = {
        'file_path': [f'sample_{i}.jpg' for i in range(100)],
        'label': np.random.choice([0, 1], 100),  # 0 = real, 1 = fake
        'confidence': np.random.uniform(0.5, 1.0, 100),
        'modality': np.random.choice(['image', 'audio', 'video'], 100)
    }
    
    df = pd.DataFrame(sample_data)
    
    # Ensure data directory exists
    os.makedirs('data', exist_ok=True)
    
    # Save sample data
    df.to_csv('data/sample_data.csv', index=False)
    print(f"Sample data created: {len(df)} records")
    return df

def run_basic_analysis(data):
    """Run basic analysis on the data"""
    print("\n=== Basic Data Analysis ===")
    print(f"Total samples: {len(data)}")
    print(f"Real samples: {len(data[data['label'] == 0])}")
    print(f"Fake samples: {len(data[data['label'] == 1])}")
    print(f"Modalities: {data['modality'].value_counts().to_dict()}")
    print(f"Average confidence: {data['confidence'].mean():.3f}")

def simulate_model_training():
    """Simulate model training process"""
    print("\n=== Simulating Model Training ===")
    models = ['CNN', 'SVM', 'Transformer', 'Bayesian', 'Vision Transformer']
    
    for model in models:
        print(f"Training {model} model...")
        # Simulate training time
        import time
        time.sleep(0.5)
        accuracy = np.random.uniform(0.75, 0.95)
        print(f"{model} training completed - Accuracy: {accuracy:.3f}")

def simulate_evaluation():
    """Simulate model evaluation"""
    print("\n=== Model Evaluation Results ===")
    models = ['CNN', 'SVM', 'Transformer', 'Bayesian', 'Vision Transformer']
    
    results = {}
    for model in models:
        accuracy = np.random.uniform(0.80, 0.95)
        precision = np.random.uniform(0.75, 0.90)
        recall = np.random.uniform(0.70, 0.88)
        f1_score = 2 * (precision * recall) / (precision + recall)
        
        results[model] = {
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1_score
        }
        
        print(f"{model}:")
        print(f"  Accuracy:  {accuracy:.3f}")
        print(f"  Precision: {precision:.3f}")
        print(f"  Recall:    {recall:.3f}")
        print(f"  F1-Score:  {f1_score:.3f}")
        print()

def main():
    """Main execution function"""
    print("=" * 60)
    print("MULTIDISCIPLINARY DEEPFAKE DETECTION - DEMO")
    print("=" * 60)
    
    # Check project structure
    if not check_project_structure():
        print("Creating missing directories...")
        os.makedirs('data', exist_ok=True)
        os.makedirs('models/saved_models', exist_ok=True)
        os.makedirs('logs', exist_ok=True)
    
    # Create and analyze sample data
    data = create_sample_data()
    run_basic_analysis(data)
    
    # Simulate training and evaluation
    simulate_model_training()
    simulate_evaluation()
    
    print("\n=== Available Project Components ===")
    print("1. Data preprocessing scripts in scripts/")
    print("2. Model implementations in src/models/")
    print("3. Training scripts in src/training/")
    print("4. Evaluation scripts in src/evaluation/")
    print("5. Jupyter notebooks in notebooks/")
    
    print("\n=== Next Steps ===")
    print("1. Add real deepfake datasets to data/raw/")
    print("2. Run: python scripts/preprocess_data.py")
    print("3. Run: python src/train.py")
    print("4. Run: python src/evaluate.py")
    print("5. Check results in logs/ and models/saved_models/")
    
    print("\n" + "=" * 60)
    print("DEMO COMPLETED SUCCESSFULLY!")
    print("=" * 60)

if __name__ == "__main__":
    main()