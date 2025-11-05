"""
Quick Test Script for Emotion Recognition Models
Tests the pipeline with a small subset of data (100 samples, 5 epochs)
"""

import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle
import os

from models import BaselineCNN, MultimodalCNN


# CONFIGURATION: Choose dataset to test
DATASET = 'crema-d'  # Change to 'meld' to test MELD dataset


def load_test_data(dataset='crema-d'):
    """Load pre-extracted test data from npz files"""
    if dataset == 'crema-d':
        npz_path = 'results/test_data.npz'
        num_classes = 6
    else:  # meld
        npz_path = 'results_meld/test_features.npz'
        num_classes = 7
    
    if not os.path.exists(npz_path):
        return None, None, None, num_classes
    
    data = np.load(npz_path)
    
    # Handle different key names in npz files
    if dataset == 'crema-d':
        X_audio = data['X_audio_test']
        X_text = data['X_text_test']
        y = data['y_test']
    else:  # meld
        X_audio = data['X_audio']
        X_text = data['X_text']
        y = data['y']
    
    return X_audio, X_text, y, num_classes


def quick_test():
    """
    Quick test with minimal data to verify everything works
    """

    dataset_name = DATASET.upper()
    num_emotions = 6 if DATASET == 'crema-d' else 7
    print(f"{dataset_name} Dataset ({num_emotions} emotions)")
    print("Testing with 100 samples and 5 epochs")   
    # Feature Extraction
    print("\nFeature Extraction")
    
    # Try to load from pre-extracted features first
    X_audio, X_text, y, num_classes = load_test_data(DATASET)
    
    if X_audio is None:
        print("No pre-extracted data found, attempting to extract from raw files")
        from feature_extraction import FeatureExtractor
        extractor = FeatureExtractor(data_path='data')
        data = extractor.extract_all_features(max_samples=100)
        
        if len(data) == 0:
            print("ERROR: No data extracted")
            return
        
        X_audio = np.array([d['audio_features'] for d in data])
        X_text = np.array([d['text_features'] for d in data])
        y = np.array([d['emotion'] for d in data])
        num_classes = 6
    else:
        # Limit to 100 samples for quick test
        if len(X_audio) > 100:
            indices = np.random.choice(len(X_audio), 100, replace=False)
            X_audio = X_audio[indices]
            X_text = X_text[indices]
            y = y[indices]
    
    print(f"Extracted features from {len(X_audio)} samples")
    
    # Data Preparation
    print("\nData Preparation")
    
    
    print(f"Audio features shape: {X_audio.shape}")
    print(f"Text features shape: {X_text.shape}")
    print(f"Emotion distribution: {np.bincount(y)}")
    
    # Split data
    min_class_count = np.min(np.bincount(y))
    use_stratify = min_class_count >= 2
    
    if use_stratify:
        X_audio_train, X_audio_test, X_text_train, X_text_test, y_train, y_test = train_test_split(
            X_audio, X_text, y, test_size=0.2, random_state=42, stratify=y
        )
    else:
        X_audio_train, X_audio_test, X_text_train, X_text_test, y_train, y_test = train_test_split(
            X_audio, X_text, y, test_size=0.2, random_state=42
        )
    
    # Use a portion of training as validation
    min_class_count_train = np.min(np.bincount(y_train))
    use_stratify_val = min_class_count_train >= 2
    
    if use_stratify_val:
        X_audio_train, X_audio_val, X_text_train, X_text_val, y_train, y_val = train_test_split(
            X_audio_train, X_text_train, y_train, test_size=0.2, random_state=42, stratify=y_train
        )
    else:
        X_audio_train, X_audio_val, X_text_train, X_text_val, y_train, y_val = train_test_split(
            X_audio_train, X_text_train, y_train, test_size=0.2, random_state=42
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
        num_classes=num_classes
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
        num_classes=num_classes
    )
    
    multimodal_history = multimodal_cnn.train(
        X_audio_train, X_text_train, y_train,
        X_audio_val, X_text_val, y_val,
        epochs=5,
        batch_size=16
    )
    
    # Evaluate multimodal
    multimodal_test_loss, multimodal_test_acc = multimodal_cnn.model.evaluate(
        [X_audio_test.reshape(X_audio_test.shape[0], X_audio_test.shape[1], 1), X_text_test],
        y_test,
        verbose=0
    )
    print(f"Multimodal Test Accuracy: {multimodal_test_acc:.4f}")
    
    print(f"\n{dataset_name} Results")
    improvement = ((multimodal_test_acc - baseline_test_acc) / baseline_test_acc) * 100 if baseline_test_acc > 0 else 0
    print(f"Baseline CNN (Audio Only):    {baseline_test_acc:.2%}")
    print(f"Multimodal CNN (Audio + Text): {multimodal_test_acc:.2%}")
    print(f"Improvement:                   {improvement:+.1f}%")

if __name__ == "__main__":
    quick_test()
