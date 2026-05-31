import os
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

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.models.cnn import CNNModel
from src.models.transformer import TransformerModel
from src.models.bayesian import BayesianModel
from src.models.vision_transformer import VisionTransformer

MODALITIES = [
    ('image', 'data/processed/processed_images.csv', 784, (1, 28, 28)),
    ('audio', 'data/processed/processed_audios.csv', 128, (1, 16, 8)),
    ('video', 'data/processed/processed_videos.csv', 1024, (1, 32, 32)),
]

results = []

for modality, csv_path, n_features, cnn_shape in MODALITIES:
    print(f"\n{'='*70}\nPROCESSING MODALITY: {modality.upper()}\n{'='*70}")
    df = pd.read_csv(csv_path)
    X = df.drop('label', axis=1).values.astype(np.float32)
    y = df['label'].values.astype(np.int64)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # CNN
    try:
        X_train_cnn = torch.tensor(X_train.reshape(-1, *cnn_shape), dtype=torch.float32)
        y_train_t = torch.tensor(y_train, dtype=torch.long)
        model = CNNModel(num_classes=2, input_channels=cnn_shape[0])
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        model.train()
        for epoch in range(5):
            optimizer.zero_grad()
            outputs = model(X_train_cnn)
            loss = criterion(outputs, y_train_t)
            loss.backward()
            optimizer.step()
        X_test_cnn = torch.tensor(X_test.reshape(-1, *cnn_shape), dtype=torch.float32)
        model.eval()
        with torch.no_grad():
            outputs = model(X_test_cnn)
            y_pred = outputs.argmax(dim=1).numpy()
        results.append({
            'modality': modality, 'model': 'CNN',
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
            'f1_score': f1_score(y_test, y_pred, average='weighted', zero_division=0)
        })
    except Exception as e:
        print(f"CNN {modality} error: {e}")

    # Transformer
    try:
        X_train_t = torch.tensor(X_train, dtype=torch.float32)
        y_train_t = torch.tensor(y_train, dtype=torch.long)
        model = TransformerModel(input_dim=n_features, model_dim=128, num_heads=2, num_layers=2, output_dim=2)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        model.train()
        for epoch in range(5):
            optimizer.zero_grad()
            outputs = model(X_train_t)
            loss = criterion(outputs, y_train_t)
            loss.backward()
            optimizer.step()
        X_test_t = torch.tensor(X_test, dtype=torch.float32)
        model.eval()
        with torch.no_grad():
            outputs = model(X_test_t)
            y_pred = outputs.argmax(dim=1).numpy()
        results.append({
            'modality': modality, 'model': 'Transformer',
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
            'f1_score': f1_score(y_test, y_pred, average='weighted', zero_division=0)
        })
    except Exception as e:
        print(f"Transformer {modality} error: {e}")

    # Sopport Vector Machine    
    try:
        model = Pipeline([
            ('scaler', StandardScaler()),
            ('svm', SVC(kernel='rbf', C=1.0, random_state=42))
        ])
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        results.append({
            'modality': modality, 'model': 'SVM',
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
            'f1_score': f1_score(y_test, y_pred, average='weighted', zero_division=0)
        })
    except Exception as e:
        print(f"SVM {modality} error: {e}")

    # Bayesian
    try:
        model = BayesianModel(prior_mean=0, prior_std=1, num_classes=2)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        results.append({
            'modality': modality, 'model': 'Bayesian',
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
            'f1_score': f1_score(y_test, y_pred, average='weighted', zero_division=0)
        })
    except Exception as e:
        print(f"Bayesian {modality} error: {e}")

    # Vision Transformer
    try:
        X_train_vit = torch.tensor(X_train.reshape(-1, 1, cnn_shape[1], cnn_shape[2]), dtype=torch.float32)
        y_train_t = torch.tensor(y_train, dtype=torch.long)
        X_train_3ch = X_train_vit.repeat(1, 3, 1, 1)
        model = VisionTransformer(img_size=cnn_shape[1], patch_size=4, num_classes=2, dim=64, depth=2, heads=2, mlp_dim=128)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        model.train()
        for epoch in range(5):
            optimizer.zero_grad()
            outputs = model(X_train_3ch)
            loss = criterion(outputs, y_train_t)
            loss.backward()
            optimizer.step()
        X_test_vit = torch.tensor(X_test.reshape(-1, 1, cnn_shape[1], cnn_shape[2]), dtype=torch.float32)
        X_test_3ch = X_test_vit.repeat(1, 3, 1, 1)
        model.eval()
        with torch.no_grad():
            outputs = model(X_test_3ch)
            y_pred = outputs.argmax(dim=1).numpy()
        results.append({
            'modality': modality, 'model': 'Vision Transformer',
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
            'f1_score': f1_score(y_test, y_pred, average='weighted', zero_division=0)
        })
    except Exception as e:
        print(f"Vision Transformer {modality} error: {e}")

# Save results
results_df = pd.DataFrame(results)
results_df.to_csv('reports/multimodal_model_evaluation_results.csv', index=False)
print("\nAll results saved to reports/multimodal_model_evaluation_results.csv")
print(results_df)
