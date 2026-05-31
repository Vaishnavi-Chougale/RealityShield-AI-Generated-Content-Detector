#!/usr/bin/env python3
"""
Minimal Training, Testing, and Evaluation Pipeline
Trains all 5 models with F1 scores - standalone execution
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

# Setup directories
os.makedirs('models/saved_models', exist_ok=True)
os.makedirs('logs', exist_ok=True)
os.makedirs('data/processed', exist_ok=True)
os.makedirs('reports', exist_ok=True)

# Add to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from src.models.cnn import CNNModel
    from src.models.transformer import TransformerModel
    from src.models.bayesian import BayesianModel
    from src.models.vision_transformer import VisionTransformer
except ImportError as e:
    print(f"Warning: Could not import all models: {e}")

class MinimalPipeline:
    def __init__(self):
        self.results = {}
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
    def generate_data(self):
        """Generate synthetic training and test data"""
        print("\n" + "="*70)
        print("PHASE 1: GENERATING SYNTHETIC DATA")
        print("="*70)
        np.random.seed(42)
        
        n_samples = 200
        X = np.random.randn(n_samples, 784).astype(np.float32)
        y = (X[:, 0] + X[:, 100] + X[:, 200] > 0).astype(np.int64)
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        print(f"✓ Generated {n_samples} total samples")
        print(f"  - Training set: {self.X_train.shape[0]} samples (784 features)")
        print(f"  - Test set: {self.X_test.shape[0]} samples (784 features)")
        print(f"  - Class distribution: {np.bincount(y)}")
        
    def train_cnn(self):
        """Train CNN model"""
        print("\n[1/5] Training CNN Model...")
        try:
            X_train = torch.tensor(self.X_train.reshape(-1, 1, 28, 28), dtype=torch.float32)
            y_train = torch.tensor(self.y_train, dtype=torch.long)
            
            model = CNNModel(num_classes=2, input_channels=1)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            criterion = nn.CrossEntropyLoss()
            
            model.train()
            for epoch in range(10):
                optimizer.zero_grad()
                outputs = model(X_train)
                loss = criterion(outputs, y_train)
                loss.backward()
                optimizer.step()
            
            # Evaluate
            X_test = torch.tensor(self.X_test.reshape(-1, 1, 28, 28), dtype=torch.float32)
            model.eval()
            with torch.no_grad():
                outputs = model(X_test)
                y_pred = outputs.argmax(dim=1).numpy()
            
            metrics = self._calculate_metrics(y_pred, 'CNN')
            torch.save(model.state_dict(), 'models/saved_models/cnn_model.pth')
            print(f"    ✓ CNN model trained and saved")
            return metrics
        except Exception as e:
            print(f"    ✗ CNN Error: {e}")
            return None
    
    def train_transformer(self):
        """Train Transformer model"""
        print("\n[2/5] Training Transformer Model...")
        try:
            X_train = torch.tensor(self.X_train, dtype=torch.float32)
            y_train = torch.tensor(self.y_train, dtype=torch.long)
            
            model = TransformerModel(
                input_dim=784,
                model_dim=256,
                num_heads=4,
                num_layers=2,
                output_dim=2
            )
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            criterion = nn.CrossEntropyLoss()
            
            model.train()
            for epoch in range(10):
                optimizer.zero_grad()
                outputs = model(X_train)
                loss = criterion(outputs, y_train)
                loss.backward()
                optimizer.step()
            
            # Evaluate
            X_test = torch.tensor(self.X_test, dtype=torch.float32)
            model.eval()
            with torch.no_grad():
                outputs = model(X_test)
                y_pred = outputs.argmax(dim=1).numpy()
            
            metrics = self._calculate_metrics(y_pred, 'Transformer')
            torch.save(model.state_dict(), 'models/saved_models/transformer_model.pth')
            print(f"    ✓ Transformer model trained and saved")
            return metrics
        except Exception as e:
            print(f"    ✗ Transformer Error: {e}")
            return None
    
    def train_svm(self):
        """Train SVM model"""
        print("\n[3/5] Training SVM Model...")
        try:
            model = Pipeline([
                ('scaler', StandardScaler()),
                ('svm', SVC(kernel='rbf', C=1.0, random_state=42))
            ])
            
            model.fit(self.X_train, self.y_train)
            y_pred = model.predict(self.X_test)
            
            metrics = self._calculate_metrics(y_pred, 'SVM')
            joblib.dump(model, 'models/saved_models/svm_model.pkl')
            print(f"    ✓ SVM model trained and saved")
            return metrics
        except Exception as e:
            print(f"    ✗ SVM Error: {e}")
            return None
    
    def train_bayesian(self):
        """Train Bayesian model"""
        print("\n[4/5] Training Bayesian Model...")
        try:
            model = BayesianModel(prior_mean=0, prior_std=1, num_classes=2)
            model.fit(self.X_train, self.y_train)
            y_pred = model.predict(self.X_test)
            
            metrics = self._calculate_metrics(y_pred, 'Bayesian')
            joblib.dump(model, 'models/saved_models/bayesian_model.pkl')
            print(f"    ✓ Bayesian model trained and saved")
            return metrics
        except Exception as e:
            print(f"    ✗ Bayesian Error: {e}")
            return None
    
    def train_vision_transformer(self):
        """Train Vision Transformer model"""
        print("\n[5/5] Training Vision Transformer Model...")
        try:
            X_train = torch.tensor(self.X_train.reshape(-1, 1, 28, 28), dtype=torch.float32)
            y_train = torch.tensor(self.y_train, dtype=torch.long)
            X_train_3ch = X_train.repeat(1, 3, 1, 1)
            
            model = VisionTransformer(
                img_size=28,
                patch_size=4,
                num_classes=2,
                dim=192,
                depth=2,
                heads=3,
                mlp_dim=512
            )
            
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            criterion = nn.CrossEntropyLoss()
            
            model.train()
            for epoch in range(10):
                optimizer.zero_grad()
                outputs = model(X_train_3ch)
                loss = criterion(outputs, y_train)
                loss.backward()
                optimizer.step()
            
            # Evaluate
            X_test = torch.tensor(self.X_test.reshape(-1, 1, 28, 28), dtype=torch.float32)
            X_test_3ch = X_test.repeat(1, 3, 1, 1)
            model.eval()
            with torch.no_grad():
                outputs = model(X_test_3ch)
                y_pred = outputs.argmax(dim=1).numpy()
            
            metrics = self._calculate_metrics(y_pred, 'Vision Transformer')
            torch.save(model.state_dict(), 'models/saved_models/vision_transformer_model.pth')
            print(f"    ✓ Vision Transformer model trained and saved")
            return metrics
        except Exception as e:
            print(f"    ✗ Vision Transformer Error: {e}")
            return None
    
    def _calculate_metrics(self, y_pred, model_name):
        """Calculate and store evaluation metrics"""
        accuracy = accuracy_score(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(self.y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(self.y_test, y_pred, average='weighted', zero_division=0)
        
        self.results[model_name] = {
            'model': model_name,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }
        
        return self.results[model_name]
    
    def save_results(self):
        """Save all results to CSV"""
        print("\n" + "="*70)
        print("PHASE 3: SAVING RESULTS")
        print("="*70)
        
        df_results = pd.DataFrame(list(self.results.values()))
        results_path = 'reports/model_evaluation_results.csv'
        df_results.to_csv(results_path, index=False)
        print(f"✓ Results saved to: {results_path}\n")
        
        print("FINAL RESULTS SUMMARY")
        print("-"*70)
        print(df_results.to_string(index=False))
        print("-"*70)
        return df_results
    
    def run(self):
        """Execute complete pipeline"""
        print("\n" + "="*70)
        print("MULTIDISCIPLINARY DEEPFAKE DETECTION - TRAINING & EVALUATION")
        print("="*70)
        
        self.generate_data()
        
        print("\n" + "="*70)
        print("PHASE 2: TRAINING ALL MODELS")
        print("="*70)
        
        self.train_cnn()
        self.train_transformer()
        self.train_svm()
        self.train_bayesian()
        self.train_vision_transformer()
        
        df_results = self.save_results()
        
        print("="*70)
        print("✓ PIPELINE COMPLETED SUCCESSFULLY")
        print("="*70)
        print(f"\nModels saved to: models/saved_models/")
        print(f"Results saved to: reports/model_evaluation_results.csv")
        print(f"Total models trained: {len(self.results)}")
        print(f"\nBest F1 Score: {df_results['f1_score'].max():.4f} ({df_results.loc[df_results['f1_score'].idxmax(), 'model']})")
        
        return df_results

if __name__ == "__main__":
    pipeline = MinimalPipeline()
    df_results = pipeline.run()
