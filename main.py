import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import joblib

# Add src to path
sys.path.append('src')

def create_sample_dataset():
    """Create sample dataset for testing"""
    np.random.seed(42)
    n_samples = 1000
    n_features = 50
    
    # Generate synthetic features
    X = np.random.randn(n_samples, n_features)
    # Create labels (0=real, 1=fake)
    y = np.random.choice([0, 1], n_samples)
    
    return X, y

def train_simple_model(X, y):
    """Train a simple SVM model"""
    print("Training SVM model...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = SVC(kernel='rbf', random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Model accuracy: {accuracy:.3f}")
    
    # Save model
    os.makedirs('models/saved_models', exist_ok=True)
    joblib.dump(model, 'models/saved_models/svm_model.pkl')
    print("Model saved to models/saved_models/svm_model.pkl")
    
    return model, accuracy

def main():
    print("=" * 50)
    print("MULTIDISCIPLINARY DEEPFAKE DETECTION")
    print("=" * 50)
    
    # Create directories
    os.makedirs('data', exist_ok=True)
    os.makedirs('models/saved_models', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    
    # Generate and train on sample data
    X, y = create_sample_dataset()
    print(f"Dataset created: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Real samples: {np.sum(y == 0)}, Fake samples: {np.sum(y == 1)}")
    
    # Train model
    model, accuracy = train_simple_model(X, y)
    
    print("\nProject successfully running!")
    print(f"Final model accuracy: {accuracy:.3f}")

    # Plagiarism checker demo
    print("\n--- Plagiarism Checker Demo ---")
    try:
        from src.models.plagiarism_checker import PlagiarismChecker
        checker = PlagiarismChecker()
        sample_text = "Artificial intelligence is transforming the world."
        report = checker.generate_report(sample_text)
        print("Plagiarism Report:")
        print(report)
    except Exception as e:
        print(f"Plagiarism checker error: {e}")

    print("\nProject structure:")
    for root, dirs, files in os.walk('.'):
        level = root.replace('.', '').count(os.sep)
        indent = ' ' * 2 * level
        print(f"{indent}{os.path.basename(root)}/")

if __name__ == "__main__":
    main()