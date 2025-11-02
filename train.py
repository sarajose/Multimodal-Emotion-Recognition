"""
Training Script for Baseline CNN vs Multimodal CNN
Trains both models and compares their performance
"""

import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle

from feature_extraction import FeatureExtractor
from models import BaselineCNN, MultimodalCNN


def train_models(data_path='data', max_samples=None):
    """
    Train both baseline and multimodal CNN models
    
    Args:
        data_path: Path to CREMA-D WAV files
        max_samples: Maximum number of samples to use
    """ 
    # Feature Extraction
    print("Feature Extraction")
    extractor = FeatureExtractor(data_path)
    data = extractor.extract_all_features(max_samples=max_samples)
    
    if len(data) == 0:
        print("No data extracted, data path incorrect")
        return
    
    # Data Preparation
    print("\n Data Preparation")

    X_audio = np.array([d['audio_features'] for d in data])
    X_text = np.array([d['text_features'] for d in data])
    y = np.array([d['emotion'] for d in data])
    
    print(f"Audio features shape: {X_audio.shape}")
    print(f"Text features shape: {X_text.shape}")
    print(f"Labels shape: {y.shape}")
    print(f"Emotion distribution: {np.bincount(y)}")
    
    # 3. Split data (80% train, 20% test)
    X_audio_train, X_audio_test, X_text_train, X_text_test, y_train, y_test = train_test_split(
        X_audio, X_text, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Split train into train and validation (80% train, 20% val)
    X_audio_train, X_audio_val, X_text_train, X_text_val, y_train, y_val = train_test_split(
        X_audio_train, X_text_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    
    print(f"\nTrain: {len(y_train)}, Val: {len(y_val)}, Test: {len(y_test)} samples")
    
    # Feature Normalization
    print("\n Feature Normalization")
    
    audio_scaler = StandardScaler()
    text_scaler = StandardScaler()
    
    X_audio_train = audio_scaler.fit_transform(X_audio_train)
    X_audio_val = audio_scaler.transform(X_audio_val)
    X_audio_test = audio_scaler.transform(X_audio_test)
    
    X_text_train = text_scaler.fit_transform(X_text_train)
    X_text_val = text_scaler.transform(X_text_val)
    X_text_test = text_scaler.transform(X_text_test)
    
    # Save serialized splits and scalers
    results_dir = Path('results')
    results_dir.mkdir(exist_ok=True)
    
    with open(results_dir / 'audio_scaler.pkl', 'wb') as f:
        pickle.dump(audio_scaler, f)
    with open(results_dir / 'text_scaler.pkl', 'wb') as f:
        pickle.dump(text_scaler, f)
    
    np.savez(results_dir / 'test_data.npz', X_audio_test=X_audio_test, X_text_test=X_text_test, y_test=y_test)
    print(" Scalers and test data saved")
    
    # Train Baseline CNN
    print("\n Training Baseline CNN (Audio only)")
    
    baseline_cnn = BaselineCNN(
        audio_dim=X_audio_train.shape[1],
        num_classes=6
    )
    
    print("\nBaseline CNN Architecture:")
    baseline_cnn.model.summary()
    
    print("\nTraining Baseline CNN...")
    baseline_history = baseline_cnn.train(
        X_audio_train, y_train,
        X_audio_val, y_val,
        epochs=50,
        batch_size=32
    )
    
    # Evaluate baseline
    baseline_train_loss, baseline_train_acc = baseline_cnn.model.evaluate(
        X_audio_train.reshape(X_audio_train.shape[0], X_audio_train.shape[1], 1),
        y_train,
        verbose=0
    )
    baseline_test_loss, baseline_test_acc = baseline_cnn.model.evaluate(
        X_audio_test.reshape(X_audio_test.shape[0], X_audio_test.shape[1], 1),
        y_test,
        verbose=0
    )
    
    print(f"\n Baseline CNN Results:")
    print(f"   Train Accuracy: {baseline_train_acc:.4f}")
    print(f"   Test Accuracy:  {baseline_test_acc:.4f}")
    
    # Save baseline model
    baseline_cnn.save(results_dir / 'baseline_cnn.h5')
    print(f"   Model saved to: results/baseline_cnn.h5")
    
    # Train Multimodal CNN
    print("\n Training Multimodal CNN (Audio + Text Fusion)")
    print("-" * 80)
    
    multimodal_cnn = MultimodalCNN(
        audio_dim=X_audio_train.shape[1],
        text_dim=X_text_train.shape[1],
        num_classes=6
    )
    
    print("\nMultimodal CNN Architecture:")
    multimodal_cnn.model.summary()
    
    print("\nTraining Multimodal CNN...")
    multimodal_history = multimodal_cnn.train(
        X_audio_train, X_text_train, y_train,
        X_audio_val, X_text_val, y_val,
        epochs=50,
        batch_size=32
    )
    
    # Evaluate multimodal
    multimodal_train_loss, multimodal_train_acc = multimodal_cnn.model.evaluate(
        [X_audio_train.reshape(X_audio_train.shape[0], X_audio_train.shape[1], 1), X_text_train],
        y_train,
        verbose=0
    )
    multimodal_test_loss, multimodal_test_acc = multimodal_cnn.model.evaluate(
        [X_audio_test.reshape(X_audio_test.shape[0], X_audio_test.shape[1], 1), X_text_test],
        y_test,
        verbose=0
    )
    
    print(f"\n Multimodal CNN Results:")
    print(f"   Train Accuracy: {multimodal_train_acc:.4f}")
    print(f"   Test Accuracy:  {multimodal_test_acc:.4f}")
    
    # Save multimodal model
    multimodal_cnn.save(results_dir / 'multimodal_cnn.h5')
    print(f"   Model saved to: results/multimodal_cnn.h5")
    
    # Summary comparison
    print("Comparison of the 2 models")
    
    improvement = ((multimodal_test_acc - baseline_test_acc) / baseline_test_acc) * 100
    print(f"\n{'Model':<25} {'Train Acc':<15} {'Test Acc':<15} {'Improvement'}")
    print(f"{'Baseline CNN (Audio)':<25} {baseline_train_acc:>6.2%}         {baseline_test_acc:>6.2%}         -")
    print(f"{'Multimodal CNN (A+T)':<25} {multimodal_train_acc:>6.2%}         {multimodal_test_acc:>6.2%}         +{improvement:.1f}%")
    
    print("\n Training complete.\n")
    
    return {
        'baseline_cnn': baseline_cnn,
        'multimodal_cnn': multimodal_cnn,
        'baseline_test_acc': baseline_test_acc,
        'multimodal_test_acc': multimodal_test_acc
    }

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train emotion recognition models')
    parser.add_argument('--data-path', type=str, default='data', 
                        help='Path to CREMA-D WAV files')
    parser.add_argument('--max-samples', type=int, default=None,
                        help='Maximum number of samples to use for testing')
    
    args = parser.parse_args()
    
    train_models(data_path=args.data_path, max_samples=args.max_samples) 