"""
Quick Test Script for Emotion Recognition Models
Tests the pipeline with a small subset of data (100 samples, 5 epochs)
"""

import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle

from feature_extraction import FeatureExtractor
from models import BaselineCNN, MultimodalCNN


def quick_test():
    """
    Quick test with minimal data to verify everything works
    """
    print("Testing with 100 samples and 5 epochs")   
    # Feature Extraction
    print("\nFeature Extraction")
    extractor = FeatureExtractor(data_path='data')
    data = extractor.extract_all_features(max_samples=100)
    
    if len(data) == 0:
        print("ERROR: No data extracted")
        return
    
    print(f"Extracted features from {len(data)} samples")
    
    # Data Preparation
    print("\nData Preparation")

    X_audio = np.array([d['audio_features'] for d in data])
    X_text = np.array([d['text_features'] for d in data])
    y = np.array([d['emotion'] for d in data])
    
    print(f"Audio features shape: {X_audio.shape}")
    print(f"Text features shape: {X_text.shape}")
    print(f"Emotion distribution: {np.bincount(y)}")
    
    # Split data
    X_audio_train, X_audio_test, X_text_train, X_text_test, y_train, y_test = train_test_split(
        X_audio, X_text, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Use a portion of training as validation
    X_audio_train, X_audio_val, X_text_train, X_text_val, y_train, y_val = train_test_split(
        X_audio_train, X_text_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    
    print(f"Train: {len(y_train)}, Val: {len(y_val)}, Test: {len(y_test)} samples")
    
    # Feature Normalization
    audio_scaler = StandardScaler()
    text_scaler = StandardScaler()
    
    X_audio_train = audio_scaler.fit_transform(X_audio_train)
    X_audio_val = audio_scaler.transform(X_audio_val)
    X_audio_test = audio_scaler.transform(X_audio_test)
    
    X_text_train = text_scaler.fit_transform(X_text_train)
    X_text_val = text_scaler.transform(X_text_val)
    X_text_test = text_scaler.transform(X_text_test)
    
    print("Features normalized")
    
    # Train Baseline CNN
    print("\nTraining Baseline CNN (Audio Only)")
    
    baseline_cnn = BaselineCNN(
        audio_dim=X_audio_train.shape[1],
        num_classes=6
    )
    
    baseline_history = baseline_cnn.train(
        X_audio_train, y_train,
        X_audio_val, y_val,
        epochs=5,  # Quick test with fewer epochs
        batch_size=16
    )
    
    # Evaluate baseline
    baseline_test_loss, baseline_test_acc = baseline_cnn.model.evaluate(
        X_audio_test.reshape(X_audio_test.shape[0], X_audio_test.shape[1], 1),
        y_test,
        verbose=0
    )
    
    print(f"Baseline Test Accuracy: {baseline_test_acc:.4f}")
    
    # Train Multimodal CNN
    print("\nTraining Multimodal CNN (Audio + Text)")
    
    multimodal_cnn = MultimodalCNN(
        audio_dim=X_audio_train.shape[1],
        text_dim=X_text_train.shape[1],
        num_classes=6
    )
    
    multimodal_history = multimodal_cnn.train(
        X_audio_train, X_text_train, y_train,
        X_audio_val, X_text_val, y_val,
        epochs=5,  # Quick test with fewer epochs
        batch_size=16
    )
    
    # Evaluate multimodal
    multimodal_test_loss, multimodal_test_acc = multimodal_cnn.model.evaluate(
        [X_audio_test.reshape(X_audio_test.shape[0], X_audio_test.shape[1], 1), X_text_test],
        y_test,
        verbose=0
    )
    print(f"Multimodal Test Accuracy: {multimodal_test_acc:.4f}")

if __name__ == "__main__":
    quick_test()
