"""
CNN Models for Emotion Recognition
- Baseline CNN: Audio-only features
- Multimodal CNN: Audio + Text fusion with attention
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model


class BaselineCNN:
    """CNN for audio only emotion recognition"""
    
    def __init__(self, audio_dim, num_classes=6):
        self.audio_dim = audio_dim
        self.num_classes = num_classes
        self.model = self._build_model()
    
    def _build_model(self):
        """CNN architecture for audio features"""
        
        # Input: audio features (47 dimensions)
        audio_input = layers.Input(shape=(self.audio_dim, 1), name='audio_input')
        
        # Layer 1: Extract patterns (64 filters)
        x = layers.Conv1D(64, kernel_size=3, padding='same', activation='relu')(audio_input)
        x = layers.MaxPooling1D(pool_size=2)(x)
        x = layers.Dropout(0.3)(x)
        
        # Layer 2: Learn higher-level features (128 filters)
        x = layers.Conv1D(128, kernel_size=3, padding='same', activation='relu')(x)
        x = layers.GlobalAveragePooling1D()(x)  # Flatten to vector
        x = layers.Dropout(0.4)(x)
        
        # Classification: Convert to emotions
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dropout(0.4)(x)
        
        # Output: 6 emotion probabilities
        output = layers.Dense(self.num_classes, activation='softmax', name='output')(x)
        
        model = Model(inputs=audio_input, outputs=output, name='BaselineCNN')
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def _reshape(self, X):
        """Reshape data for Conv1D"""
        return X.reshape(X.shape[0], X.shape[1], 1)
    
    def train(self, X_train, y_train, X_val, y_val, epochs=50, batch_size=32):
        """Train the baseline CNN model"""
        callbacks = [
            keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
        ]
        
        history = self.model.fit(
            self._reshape(X_train), y_train,
            validation_data=(self._reshape(X_val), y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        return history
    
    def predict(self, X):
        """Make predictions"""
        return self.model.predict(self._reshape(X))
    
    def save(self, filepath):
        """Save model"""
        self.model.save(filepath)
    
    def load(self, filepath):
        """Load model"""
        self.model = keras.models.load_model(filepath)


class MultimodalCNN:
    """Multimodal CNN - combines audio + text with simple concatenation"""
    
    def __init__(self, audio_dim, text_dim, num_classes=6):
        self.audio_dim = audio_dim
        self.text_dim = text_dim
        self.num_classes = num_classes
        self.model = self._build_model()
    
    def _build_model(self):
        """Build SIMPLIFIED multimodal CNN"""
        
        # AUDIO 
        audio_input = layers.Input(shape=(self.audio_dim, 1), name='audio_input')
        
        # Extract audio patterns
        a = layers.Conv1D(64, kernel_size=3, padding='same', activation='relu')(audio_input)
        a = layers.MaxPooling1D(pool_size=2)(a)
        a = layers.Conv1D(128, kernel_size=3, padding='same', activation='relu')(a)
        a = layers.GlobalAveragePooling1D()(a)
        audio_features = layers.Dropout(0.3)(a)
        
        # TEXT
        text_input = layers.Input(shape=(self.text_dim,), name='text_input')
        
        # Expand text features to match audio size
        t = layers.Dense(64, activation='relu')(text_input)
        text_features = layers.Dense(128, activation='relu')(t) 
        
        # Stack audio and text together
        combined = layers.Concatenate()([audio_features, text_features])  # 128 + 128 = 256
        combined = layers.Dropout(0.4)(combined)
        
        # Predict emotion
        x = layers.Dense(64, activation='relu')(combined)
        output = layers.Dense(self.num_classes, activation='softmax')(x)
        
        model = Model(
            inputs=[audio_input, text_input],
            outputs=output,
            name='MultimodalCNN'
        )
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train(self, X_audio_train, X_text_train, y_train, 
              X_audio_val, X_text_val, y_val, 
              epochs=50, batch_size=32):
        """Train the multimodal CNN model"""
        
        # Reshape audio for Conv1D
        X_audio_train_reshaped = X_audio_train.reshape(X_audio_train.shape[0], X_audio_train.shape[1], 1)
        X_audio_val_reshaped = X_audio_val.reshape(X_audio_val.shape[0], X_audio_val.shape[1], 1)
        
        # Callbacks
        early_stop = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        
        reduce_lr = keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6
        )
        
        # Train
        history = self.model.fit(
            [X_audio_train_reshaped, X_text_train],
            y_train,
            validation_data=([X_audio_val_reshaped, X_text_val], y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stop, reduce_lr],
            verbose=1
        )
        
        return history
    
    def predict(self, X_audio, X_text):
        """Make predictions"""
        X_audio_reshaped = X_audio.reshape(X_audio.shape[0], X_audio.shape[1], 1)
        return self.model.predict([X_audio_reshaped, X_text])
    
    def save(self, filepath):
        self.model.save(filepath)
    
    def load(self, filepath):
        self.model = keras.models.load_model(filepath)

if __name__ == "__main__":
    print("Testing Baseline CNN architecture...")
    baseline = BaselineCNN(audio_dim=47, num_classes=6)  # 47 audio features
    baseline.model.summary()
    
    print("Testing Multimodal CNN architecture")
    multimodal = MultimodalCNN(audio_dim=47, text_dim=5, num_classes=6)  # 47 audio + 5 text
    multimodal.model.summary()
