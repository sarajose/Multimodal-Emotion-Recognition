"""
Training Script for MELD Dataset
Trains baseline and multimodal CNN models on MELD features
"""
import os
import argparse
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt

# Import models from existing models.py
from models import BaselineCNN, MultimodalCNN

def load_meld_data(data_path='results_meld'):
    """Load MELD features from saved files"""
    
    # Load train, dev, test splits
    print("Loading MELD data")
    
    train_data = np.load(os.path.join(data_path, 'train_features.npz'))
    dev_data = np.load(os.path.join(data_path, 'dev_features.npz'))
    test_data = np.load(os.path.join(data_path, 'test_features.npz'))
    
    # Extract features
    X_audio_train = train_data['X_audio']
    X_text_train = train_data['X_text']
    y_train = train_data['y']
    
    X_audio_dev = dev_data['X_audio']
    X_text_dev = dev_data['X_text']
    y_dev = dev_data['y']
    
    X_audio_test = test_data['X_audio']
    X_text_test = test_data['X_text']
    y_test = test_data['y']
    
    print(f"\nDataset loaded:")
    print(f"Train: {len(y_train)} samples")
    print(f"Dev: {len(y_dev)} samples")
    print(f"Test: {len(y_test)} samples")
    print(f"Audio features: {X_audio_train.shape[1]} dimensions")
    print(f"Text features: {X_text_train.shape[1]} dimensions")
    print(f"Classes: {len(np.unique(y_train))} emotions")
    
    # Emotion distribution
    print(f"\nTrain emotion distribution:")
    for i in range(len(np.unique(y_train))):
        count = np.sum(y_train == i)
        print(f"  Class {i}: {count} samples ({count/len(y_train)*100:.1f}%)")
    
    return (X_audio_train, X_text_train, y_train), \
           (X_audio_dev, X_text_dev, y_dev), \
           (X_audio_test, X_text_test, y_test)


def normalize_features(X_audio_train, X_audio_dev, X_audio_test,
                       X_text_train, X_text_dev, X_text_test):
    """Normalize audio and text features using StandardScaler"""
    
    print("\nNormalizing features")
    # Audio features
    audio_scaler = StandardScaler()
    X_audio_train_norm = audio_scaler.fit_transform(X_audio_train)
    X_audio_dev_norm = audio_scaler.transform(X_audio_dev)
    X_audio_test_norm = audio_scaler.transform(X_audio_test)
    
    # Text features
    text_scaler = StandardScaler()
    X_text_train_norm = text_scaler.fit_transform(X_text_train)
    X_text_dev_norm = text_scaler.transform(X_text_dev)
    X_text_test_norm = text_scaler.transform(X_text_test)
    
    return (X_audio_train_norm, X_audio_dev_norm, X_audio_test_norm,
            X_text_train_norm, X_text_dev_norm, X_text_test_norm,
            audio_scaler, text_scaler)


def train_model(model, X_train, y_train, X_val, y_val, model_name, 
                output_path, epochs=50, class_weights=None):
    """Train a model with early stopping"""
    print(f"Training {model_name}")

    
    # Callbacks
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy', 
        patience=15,
        restore_best_weights=True,
        mode='max',
        verbose=1
    )
    
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=7,
        min_lr=1e-7,
        verbose=1 
    )
    
    # Train
    keras_model = model.model if hasattr(model, 'model') else model
    
    history = keras_model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=64,
        callbacks=[early_stop, reduce_lr],
        class_weight=class_weights,
        verbose=1
    )
    
    # Save model
    model_file = os.path.join(output_path, f'{model_name.lower().replace(" ", "_")}.h5')
    keras_model.save(model_file)
    print(f"\nModel saved to {model_file}")
    
    # Plot training history
    plot_training_history(history, model_name, output_path)
    
    return model, history


def plot_training_history(history, model_name, output_path):
    """Plot and save training history"""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    # Accuracy
    ax1.plot(history.history['accuracy'], label='Train')
    ax1.plot(history.history['val_accuracy'], label='Validation')
    ax1.set_title(f'{model_name} - Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Loss
    ax2.plot(history.history['loss'], label='Train')
    ax2.plot(history.history['val_loss'], label='Validation')
    ax2.set_title(f'{model_name} - Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_file = os.path.join(output_path, f'{model_name.lower().replace(" ", "_")}_history.png')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Training history plot saved to {plot_file}")


def evaluate_model(model, X_test, y_test, model_name):
    """Evaluate model on test set"""
    print(f"Evaluating {model_name}")
    
    # Evaluate
    keras_model = model.model if hasattr(model, 'model') else model
    test_loss, test_acc = keras_model.evaluate(X_test, y_test, verbose=0)
    
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
    return test_loss, test_acc


def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description='Train models on MELD dataset')
    parser.add_argument('--data-path', type=str, default='results_meld',
                       help='Path to MELD features')
    parser.add_argument('--output-path', type=str, default='results_meld',
                       help='Path to save models and results')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Maximum number of epochs')
    parser.add_argument('--use-class-weights', action='store_true',
                       help='Use class weights for imbalanced data')
    args = parser.parse_args()
    
    # Output directory
    os.makedirs(args.output_path, exist_ok=True)
    
    # Load data
    (X_audio_train, X_text_train, y_train), \
    (X_audio_dev, X_text_dev, y_dev), \
    (X_audio_test, X_text_test, y_test) = load_meld_data(args.data_path)
    
    # Normalize features
    X_audio_train_norm, X_audio_dev_norm, X_audio_test_norm, \
    X_text_train_norm, X_text_dev_norm, X_text_test_norm, \
    audio_scaler, text_scaler = normalize_features(
        X_audio_train, X_audio_dev, X_audio_test,
        X_text_train, X_text_dev, X_text_test
    )
    
    # Reshape for CNN
    X_audio_train_cnn = X_audio_train_norm.reshape(-1, X_audio_train_norm.shape[1], 1)
    X_audio_dev_cnn = X_audio_dev_norm.reshape(-1, X_audio_dev_norm.shape[1], 1)
    X_audio_test_cnn = X_audio_test_norm.reshape(-1, X_audio_test_norm.shape[1], 1)
    
    # Compute class weights
    class_weights = None
    if args.use_class_weights:
        print("\nComputing class weights for imbalanced data")
        classes = np.unique(y_train)
        weights = compute_class_weight('balanced', classes=classes, y=y_train)
        class_weights = dict(zip(classes, weights))
        print("Class weights:", class_weights)
    
    # Number of classes
    num_classes = len(np.unique(y_train))
    
    # Train baseline CNN
    print("Baseline CNN")

    
    baseline_model = BaselineCNN(
        audio_dim=X_audio_train_cnn.shape[1],
        num_classes=num_classes
    )
    
    baseline_model, baseline_history = train_model(
        model=baseline_model,
        X_train=X_audio_train_cnn,
        y_train=y_train,
        X_val=X_audio_dev_cnn,
        y_val=y_dev,
        model_name='Baseline CNN',
        output_path=args.output_path,
        epochs=args.epochs,
        class_weights=class_weights
    )
    
    # Evaluate Baseline
    baseline_test_loss, baseline_test_acc = evaluate_model(
        baseline_model,
        X_audio_test_cnn,
        y_test,
        'Baseline CNN'
    )
    
    # Train multimodal CNN
    print("Multimodal CNN (Audio and Text)")
    
    multimodal_model = MultimodalCNN(
        audio_dim=X_audio_train_cnn.shape[1],
        text_dim=X_text_train_norm.shape[1],
        num_classes=num_classes
    )
    
    multimodal_model, multimodal_history = train_model(
        model=multimodal_model,
        X_train=[X_audio_train_cnn, X_text_train_norm],
        y_train=y_train,
        X_val=[X_audio_dev_cnn, X_text_dev_norm],
        y_val=y_dev,
        model_name='Multimodal CNN',
        output_path=args.output_path,
        epochs=args.epochs,
        class_weights=class_weights
    )
    
    # Evaluate Multimodal
    multimodal_test_loss, multimodal_test_acc = evaluate_model(
        multimodal_model,
        [X_audio_test_cnn, X_text_test_norm],
        y_test,
        'Multimodal CNN'
    )
    
    # Save test data for evaluation
    test_data_file = os.path.join(args.output_path, 'test_data.npz')
    np.savez(
        test_data_file,
        X_audio_test=X_audio_test_cnn,
        X_text_test=X_text_test_norm,
        y_test=y_test
    )
    print(f"\nTest data saved to {test_data_file}")  

if __name__ == '__main__':
    main()