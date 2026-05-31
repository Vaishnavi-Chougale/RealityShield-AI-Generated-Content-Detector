#!/usr/bin/env python3
"""
Comprehensive Training, Testing, and Evaluation Pipeline
Trains all 5 models (CNN, Transformer, SVM, Bayesian, Vision Transformer)
Evaluates with F1 scores, accuracy, precision, and recall
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import Config
from src.models.cnn import CNNModel
from src.models.transformer import TransformerModel
from src.models.svm import SVMModel
from src.models.bayesian import BayesianModel
from src.models.vision_transformer import VisionTransformer
from src.utils.logger import setup_logger

# Setup
Config.ensure_directories()
logger = setup_logger(__name__, os.path.join(Config.LOG_DIR, 'complete_pipeline.log'))

class CompletePipeline:
    def __init__(self):
        """Initialize complete training/evaluation pipeline"""
        self.train_data_path = os.path.join(Config.PROCESSED_DATA_DIR, 'train_data.csv')
        self.test_data_path = os.path.join(Config.PROCESSED_DATA_DIR, 'test_data.csv')
        self.results = {}
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
    def generate_synthetic_data(self):
        """Generate synthetic training and test data"""
        logger.info("Generating synthetic data...")
        np.random.seed(42)
        
        # Generate 200 samples with 784 features (flattened 28x28 image-like)
        n_samples = 200
        X = np.random.randn(n_samples, 784).astype(np.float32)
        y = (X[:, 0] + X[:, 100] + X[:, 200] > 0).astype(np.int64)
        
        # 80/20 train/test split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        logger.info(f"Generated data: Train {self.X_train.shape}, Test {self.X_test.shape}")
        print(f"✓ Generated synthetic data: {self.X_train.shape[0]} train, {self.X_test.shape[0]} test samples")
        
    def train_cnn(self):
        """Train CNN model"""
        logger.info("Training CNN model...")
        try:
            X_train = torch.tensor(self.X_train.reshape(-1, 1, 28, 28), dtype=torch.float32)
            y_train = torch.tensor(self.y_train, dtype=torch.long)
            
            model = CNNModel(num_classes=2, input_channels=1)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            criterion = nn.CrossEntropyLoss()
            
            model.train()
            for epoch in range(10):  # Reduced epochs for demo
                optimizer.zero_grad()
                outputs = model(X_train)
                loss = criterion(outputs, y_train)
                loss.backward()
                optimizer.step()
                if (epoch + 1) % 5 == 0:
                    logger.info(f"CNN Epoch {epoch+1}, Loss: {loss.item():.4f}")
            
            # Evaluate
            X_test = torch.tensor(self.X_test.reshape(-1, 1, 28, 28), dtype=torch.float32)
            model.eval()
            with torch.no_grad():
                outputs = model(X_test)
                y_pred = outputs.argmax(dim=1).numpy()
            
            metrics = self._calculate_metrics(y_pred, 'CNN')
            
            # Save
            model_path = os.path.join(Config.MODEL_DIR, 'cnn_model.pth')
            torch.save(model.state_dict(), model_path)
            logger.info(f"CNN model saved to {model_path}")
            
            return metrics
        except Exception as e:
            logger.error(f"Error training CNN: {e}", exc_info=True)
            return None
    
    def train_transformer(self):
        """Train Transformer model"""
        logger.info("Training Transformer model...")
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
                if (epoch + 1) % 5 == 0:
                    logger.info(f"Transformer Epoch {epoch+1}, Loss: {loss.item():.4f}")
            
            # Evaluate
            X_test = torch.tensor(self.X_test, dtype=torch.float32)
            model.eval()
            with torch.no_grad():
                outputs = model(X_test)
                y_pred = outputs.argmax(dim=1).numpy()
            
            metrics = self._calculate_metrics(y_pred, 'Transformer')
            
            # Save
            model_path = os.path.join(Config.MODEL_DIR, 'transformer_model.pth')
            torch.save(model.state_dict(), model_path)
            logger.info(f"Transformer model saved to {model_path}")
            
            return metrics
        except Exception as e:
            logger.error(f"Error training Transformer: {e}", exc_info=True)
            return None
    
    def train_svm(self):
        """Train SVM model"""
        logger.info("Training SVM model...")
        try:
            from sklearn.svm import SVC
            from sklearn.preprocessing import StandardScaler
            from sklearn.pipeline import Pipeline
            
            model = Pipeline([
                ('scaler', StandardScaler()),
                ('svm', SVC(kernel='rbf', C=1.0))
            ])
            
            model.fit(self.X_train, self.y_train)
            y_pred = model.predict(self.X_test)
            
            metrics = self._calculate_metrics(y_pred, 'SVM')
            
            # Save
            model_path = os.path.join(Config.MODEL_DIR, 'svm_model.pkl')
            joblib.dump(model, model_path)
            logger.info(f"SVM model saved to {model_path}")
            
            return metrics
        except Exception as e:
            logger.error(f"Error training SVM: {e}", exc_info=True)
            return None
    
    def train_bayesian(self):
        """Train Bayesian model"""
        logger.info("Training Bayesian model...")
        try:
            model = BayesianModel(prior_mean=0, prior_std=1, num_classes=2)
            model.fit(self.X_train, self.y_train)
            y_pred = model.predict(self.X_test)
            
            metrics = self._calculate_metrics(y_pred, 'Bayesian')
            
            # Save
            model_path = os.path.join(Config.MODEL_DIR, 'bayesian_model.pkl')
            joblib.dump(model, model_path)
            logger.info(f"Bayesian model saved to {model_path}")
            
            return metrics
        except Exception as e:
            logger.error(f"Error training Bayesian: {e}", exc_info=True)
            return None
    
    def train_vision_transformer(self):
        """Train Vision Transformer model"""
        logger.info("Training Vision Transformer model...")
        try:
            X_train = torch.tensor(self.X_train.reshape(-1, 1, 28, 28), dtype=torch.float32)
            y_train = torch.tensor(self.y_train, dtype=torch.long)
            
            # Repeat to create 3-channel input
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
                if (epoch + 1) % 5 == 0:
                    logger.info(f"ViT Epoch {epoch+1}, Loss: {loss.item():.4f}")
            
            # Evaluate
            X_test = torch.tensor(self.X_test.reshape(-1, 1, 28, 28), dtype=torch.float32)
            X_test_3ch = X_test.repeat(1, 3, 1, 1)
            model.eval()
            with torch.no_grad():
                outputs = model(X_test_3ch)
                y_pred = outputs.argmax(dim=1).numpy()
            
            metrics = self._calculate_metrics(y_pred, 'Vision Transformer')
            
            # Save
            model_path = os.path.join(Config.MODEL_DIR, 'vision_transformer_model.pth')
            torch.save(model.state_dict(), model_path)
            logger.info(f"Vision Transformer model saved to {model_path}")
            
            return metrics
        except Exception as e:
            logger.error(f"Error training Vision Transformer: {e}", exc_info=True)
            return None
    
    def _calculate_metrics(self, y_pred, model_name):
        """Calculate evaluation metrics"""
        accuracy = accuracy_score(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(self.y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(self.y_test, y_pred, average='weighted', zero_division=0)
        
        metrics = {
            'model': model_name,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }
        
        self.results[model_name] = metrics
        
        print(f"\n{'='*60}")
        print(f"{model_name} Results")
        print(f"{'='*60}")
        print(f"Accuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1 Score:  {f1:.4f}")
        print(f"{'='*60}")
        
        return metrics
    
    def save_results(self):
        """Save all results to CSV"""
        df_results = pd.DataFrame(list(self.results.values()))
        results_path = os.path.join(Config.REPORT_DIR, 'model_evaluation_results.csv')
        df_results.to_csv(results_path, index=False)
        logger.info(f"Results saved to {results_path}")
        
        print(f"\n{'='*60}")
        print("FINAL RESULTS SUMMARY")
        print(f"{'='*60}")
        print(df_results.to_string(index=False))
        print(f"{'='*60}")
    
    def run_pipeline(self):
        """Execute complete pipeline"""
        print("\n" + "="*60)
        print("MULTIDISCIPLINARY DEEPFAKE DETECTION - FULL PIPELINE")
        print("="*60)
        
        # Phase 1: Data generation
        self.generate_synthetic_data()
        
        # Phase 2: Train all models
        print("\n" + "="*60)
        print("TRAINING ALL MODELS")
        print("="*60)
        
        self.train_cnn()
        self.train_transformer()
        self.train_svm()
        self.train_bayesian()
        self.train_vision_transformer()
        
        # Phase 3: Save results
        self.save_results()
        
        print("\n✓ Pipeline completed successfully!")
        print(f"Models saved to: {Config.MODEL_DIR}")
        print(f"Results saved to: {os.path.join(Config.REPORT_DIR, 'model_evaluation_results.csv')}")

if __name__ == "__main__":
    pipeline = CompletePipeline()
    pipeline.run_pipeline()
