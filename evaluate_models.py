import os
import numpy as np
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

def load_test_data():
    """Load test data"""
    np.random.seed(123)  # Different seed for test data
    n_samples = 200
    X = np.random.randn(n_samples, 144)  # Match training data features
    y = (X[:, 0] + X[:, 64] + X[:, 96] > 0).astype(int)
    return X, y

def evaluate_model(model_path, X_test, y_test):
    """Evaluate a single model"""
    if not os.path.exists(model_path):
        return None
    
    model = joblib.load(model_path)
    y_pred = model.predict(X_test)
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred)
    }
    
    return metrics

def main():
    print("=" * 60)
    print("DEEPFAKE DETECTION - MODEL EVALUATION")
    print("=" * 60)
    
    # Load test data
    X_test, y_test = load_test_data()
    print(f"Test dataset: {X_test.shape[0]} samples")
    print(f"Real: {np.sum(y_test == 0)}, Fake: {np.sum(y_test == 1)}")
    
    # Model paths
    models = {
        'SVM': 'models/saved_models/svm_model.pkl',
        'RandomForest': 'models/saved_models/randomforest_model.pkl', 
        'LogisticRegression': 'models/saved_models/logisticregression_model.pkl'
    }
    
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS:")
    print("=" * 60)
    
    all_results = {}
    
    for name, path in models.items():
        print(f"\n{name}:")
        print("-" * 40)
        
        results = evaluate_model(path, X_test, y_test)
        if results:
            all_results[name] = results
            for metric, value in results.items():
                print(f"{metric.capitalize():12}: {value:.3f}")
        else:
            print("Model not found!")
    
    # Find best model
    if all_results:
        best_model = max(all_results, key=lambda x: all_results[x]['f1_score'])
        best_f1 = all_results[best_model]['f1_score']
        
        print("\n" + "=" * 60)
        print(f"BEST MODEL: {best_model} (F1-Score: {best_f1:.3f})")
        print("=" * 60)

if __name__ == "__main__":
    main()