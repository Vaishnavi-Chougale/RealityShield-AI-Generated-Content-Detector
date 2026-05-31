import os
import sys
import argparse
import numpy as np
import pandas as pd
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib
from pathlib import Path
from tqdm import tqdm

# Import CNN model from src
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.models.cnn import CNNModel

class VideoFrameDataset(Dataset):
    """Dataset for loading video frames"""
    def __init__(self, video_dir, labels_file=None, frame_sample_rate=10, transform=None):
        self.video_dir = Path(video_dir)
        self.frame_sample_rate = frame_sample_rate
        self.transform = transform
        self.samples = []
        
        # Get all video files
        video_files = list(self.video_dir.glob('*.mp4')) + list(self.video_dir.glob('*.avi'))
        
        if not video_files:
            print(f"Warning: No video files found in {video_dir}")
            # Create dummy data for testing
            self.samples = [(None, 0)] * 100  # 100 dummy samples
            self.dummy_mode = True
        else:
            self.dummy_mode = False
            # Process each video
            for video_path in video_files:
                # Assign label based on filename (you can modify this logic)
                # Assume files with 'fake' in name are fake (label=1), others are real (label=0)
                label = 1 if 'fake' in video_path.stem.lower() else 0
                
                # Extract frames from video
                frames = self.extract_frames(str(video_path))
                for frame in frames:
                    self.samples.append((frame, label))
    
    def extract_frames(self, video_path, max_frames=10):
        """Extract frames from video"""
        frames = []
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"Warning: Could not open video {video_path}")
            return frames
            
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        sample_indices = np.linspace(0, frame_count-1, min(max_frames, frame_count), dtype=int)
        
        for idx in sample_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                # Resize frame to standard size
                frame = cv2.resize(frame, (224, 224))
                # Convert BGR to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
        
        cap.release()
        return frames
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        if self.dummy_mode:
            # Return dummy data
            frame = np.random.randn(3, 224, 224).astype(np.float32)
            label = np.random.randint(0, 2)
            return torch.tensor(frame), torch.tensor(label, dtype=torch.long)
        
        frame, label = self.samples[idx]
        
        # Convert to tensor and normalize
        frame = frame.astype(np.float32) / 255.0
        frame = np.transpose(frame, (2, 0, 1))  # HWC to CHW
        
        return torch.tensor(frame), torch.tensor(label, dtype=torch.long)

def train_cnn_model(data_dir='data/processed/processed_videos', output_path='models/saved_models/cnn_model.pth', 
                     epochs=10, batch_size=16, learning_rate=0.001):
    """Train CNN model for video deepfake detection"""
    print("\n" + "="*60)
    print("TRAINING CNN FOR VIDEO DEEPFAKE DETECTION")
    print("="*60)
    
    # Create dataset
    dataset = VideoFrameDataset(data_dir)
    print(f"Loaded {len(dataset)} frames from videos")
    
    # Split dataset
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CNNModel(num_classes=2, input_channels=3).to(device)
    print(f"Using device: {device}")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    print("\nStarting training...")
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')
        for frames, labels in progress_bar:
            frames, labels = frames.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(frames)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
            progress_bar.set_postfix({'loss': loss.item(), 'acc': 100.*train_correct/train_total})
        
        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for frames, labels in val_loader:
                frames, labels = frames.to(device), labels.to(device)
                outputs = model(frames)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        train_acc = 100 * train_correct / train_total
        val_acc = 100 * val_correct / val_total
        avg_loss = train_loss / len(train_loader)
        
        print(f"Epoch [{epoch+1}/{epochs}] - Loss: {avg_loss:.4f}, Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%")
    
    # Save model
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(model.state_dict(), output_path)
    print(f"\nCNN model saved to {output_path}")
    
    # --- Evaluation metrics on validation set ---
    from sklearn.metrics import classification_report, confusion_matrix
    y_true = []
    y_pred = []
    model.eval()
    with torch.no_grad():
        for frames, labels in val_loader:
            frames, labels = frames.to(device), labels.to(device)
            outputs = model(frames)
            _, predicted = torch.max(outputs.data, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
    print("\nValidation Classification Report:")
    unique_labels = sorted(set(y_true + y_pred))
    label_map = {0: 'Real', 1: 'Fake'}
    target_names = [label_map[l] for l in unique_labels if l in label_map]
    try:
        print(classification_report(y_true, y_pred, target_names=target_names, labels=unique_labels))
    except ValueError as e:
        print(f"[Warning] Could not compute full classification report: {e}")
        print(classification_report(y_true, y_pred, labels=unique_labels))
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred, labels=unique_labels))
    return model

def create_multimodal_dataset():
    """Create sample multimodal dataset"""
    np.random.seed(42)
    n_samples = 500
    
    # Image features (simulated)
    image_features = np.random.randn(n_samples, 64)
    
    # Audio features (simulated) 
    audio_features = np.random.randn(n_samples, 32)
    
    # Video features (simulated)
    video_features = np.random.randn(n_samples, 48)
    
    # Combine all features
    X = np.hstack([image_features, audio_features, video_features])
    
    # Create labels with some correlation to features
    y = (X[:, 0] + X[:, 64] + X[:, 96] > 0).astype(int)
    
    return X, y

def train_all_models(X, y):
    """Train multiple models"""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    models = {
        'SVM': SVC(kernel='rbf', random_state=42),
        'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
        'LogisticRegression': LogisticRegression(random_state=42)
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"{name} Accuracy: {accuracy:.3f}")
        
        # Save model
        model_path = f'models/saved_models/{name.lower()}_model.pkl'
        joblib.dump(model, model_path)
        print(f"Saved to {model_path}")
        
        results[name] = accuracy
    
    return results

def main():
    parser = argparse.ArgumentParser(description='Train models for deepfake detection')
    parser.add_argument('--model', type=str, default='all', 
                        choices=['cnn', 'svm', 'rf', 'lr', 'all'],
                        help='Model to train (cnn, svm, rf, lr, or all)')
    parser.add_argument('--data_dir', type=str, default='data/processed/processed_videos',
                        help='Directory containing video data for CNN')
    parser.add_argument('--output', type=str, default='models/saved_models',
                        help='Output directory for saved models')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of epochs for CNN training')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for CNN training')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Learning rate for CNN training')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("DEEPFAKE DETECTION - MODEL TRAINING")
    print("=" * 60)
    
    if args.model == 'cnn':
        # Train CNN model for video deepfake detection
        output_path = os.path.join(args.output, 'cnn_model.pth')
        train_cnn_model(
            data_dir=args.data_dir,
            output_path=output_path,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate
        )
    else:
        # Train traditional ML models
        X, y = create_multimodal_dataset()
        print(f"\nDataset: {X.shape[0]} samples, {X.shape[1]} features")
        print(f"Real: {np.sum(y == 0)}, Fake: {np.sum(y == 1)}")
        
        if args.model == 'all':
            # Train all traditional models
            results = train_all_models(X, y)
            
            # Also train CNN if requested
            print("\nTraining CNN model...")
            output_path = os.path.join(args.output, 'cnn_model.pth')
            train_cnn_model(
                data_dir=args.data_dir,
                output_path=output_path,
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate
            )
            
            print("\n" + "=" * 60)
            print("TRAINING RESULTS (Traditional Models):")
            print("=" * 60)
            for model, accuracy in results.items():
                print(f"{model:20}: {accuracy:.3f}")
            
            print(f"\nBest traditional model: {max(results, key=results.get)} ({max(results.values()):.3f})")
        else:
            # Train specific traditional model
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            if args.model == 'svm':
                model = SVC(kernel='rbf', random_state=42)
                model_name = 'SVM'
            elif args.model == 'rf':
                model = RandomForestClassifier(n_estimators=100, random_state=42)
                model_name = 'RandomForest'
            elif args.model == 'lr':
                model = LogisticRegression(random_state=42)
                model_name = 'LogisticRegression'
            
            print(f"\nTraining {model_name}...")
            model.fit(X_train, y_train)
            
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            print(f"{model_name} Accuracy: {accuracy:.3f}")
            
            # Save model
            model_path = os.path.join(args.output, f'{model_name.lower()}_model.pkl')
            os.makedirs(args.output, exist_ok=True)
            joblib.dump(model, model_path)
            print(f"Saved to {model_path}")

if __name__ == "__main__":
    main()
