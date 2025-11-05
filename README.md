# Emotion Recognition: Baseline vs Multimodal CNN

A comparison of **audio-only** vs **multimodal (audio+text)** CNN architectures for emotion recognition on the CREMA-D dataset.

## Overview

This project compares two deep learning (CNN) approaches for speech emotion recognition:
1. **Baseline CNN**: Audio features only
2. **Multimodal CNN**: Audio + Text features

## Dataset

### CREMA-D (Crowd-sourced Emotional Multimodal Actors Dataset)
- 7,442 audio clips (.wav files)
- 6 emotions: Angry, Disgust, Fear, Happy, Neutral, Sad
- 91 actors (48 male, 43 female)

**Download:**
1. Download from [Kaggle - CREMA-D Dataset](https://www.kaggle.com/datasets/ejlok1/cremad)
2. Create a `data/` folder and extract `.wav` files into it.

### MELD (Multimodal EmotionLines Dataset)
- Multimodal emotion recognition dataset
- 7 emotions: anger, disgust, fear, joy, neutral, sadness, surprise

**Download:**
1. Download from [Kaggle - MELD Dataset](https://www.kaggle.com/datasets/zaber666/meld-dataset)
2. Create a `MELD_raw_data/` folder and extract `MELD.Raw` folder into it.

**Note:** Both data folders are not included in this repository due to size constraints.

## Project Structure

```
project/
├── feature_extraction.py       # Audio and text feature extraction (CREMA-D)
├── feature_extraction_meld.py  # Audio and text feature extraction (MELD)
├── models.py                   # CNN architectures
├── train.py                    # Training pipeline (CREMA-D)
├── train_meld.py               # Training pipeline (MELD)
├── evaluate.py                 # Model evaluation (CREMA-D)
├── evaluate_meld.py            # Model evaluation (MELD)
├── quick_test.py               # Quick test script
├── requirements.txt            # Dependencies
├── results/                    # CREMA-D models and outputs
├── results_meld/               # MELD models and outputs
├── figures/                    # CREMA-D visualizations
└── figures_meld/               # MELD visualizations
```

## Usage

### Quick Test

To verify the installation:

```bash
python quick_test.py
```

This will:
- Load pre-extracted features from both CREMA-D and MELD datasets (100 samples each)
- Train both models for 5 epochs

Note: The test data is already included in `results/` and `results_meld/` folders.

### Full Training

```bash
# Train models on full dataset
python train.py --data-path data

# For MELD dataset
python train_meld.py --data-path MELD_raw_data
```

### Evaluation

```bash
# Evaluate CREMA-D models
python evaluate.py

# Evaluate MELD models
python evaluate_meld.py
```

This will:
1. Load trained models and test data
2. Generate predictions and calculate metrics
3. Create and save visualizations (confusion matrices, ROC curves, etc.)

## Model Architectures

### Baseline CNN (audio only)
```
Input (47, 1) 
→ Conv1D(128) + BatchNorm + MaxPool(2) + Dropout(0.3)
→ Conv1D(256) + BatchNorm + GlobalAvgPool + Dropout(0.4)
→ Dense(128, relu) + Dropout(0.5)
→ Dense(6/7, softmax) [Output]
```
**Architecture Details:**
- **Enhanced architecture** with increased capacity for better performance
- 2 convolutional layers (128 → 256 filters) to extract audio patterns
- **Batch Normalization** added after each Conv1D layer for training stability
- MaxPooling and Dropout for regularization
- 1 dense hidden layer (128 units, increased from 64)
- Higher dropout (0.5) for better generalization
- Reduced learning rate (0.0005) for better convergence
- Output: 6 classes (CREMA-D) or 7 classes (MELD)

### Multimodal CNN (audio + text)
```
Audio Branch: 
  Input (47, 1) 
  → Conv1D(128) + BatchNorm + MaxPool(2)
  → Conv1D(256) + BatchNorm + GlobalAvgPool
  → Dropout(0.3) → [256 features]

Text Branch:  
  Input (5,) 
  → Dense(128, relu) + BatchNorm
  → Dense(256, relu) → [256 features]

Fusion: 
  Concatenate [audio + text] → (512,)
  → Dropout(0.4)
  → Dense(128, relu)
  → Dropout(0.5)
  → Dense(6/7, softmax) [Output]
```
**Architecture Details:**
- **Enhanced architecture** with significantly increased capacity
- Audio and text branches process features separately
- **Batch Normalization** added for training stability
- Audio branch: 128 → 256 filters (increased from 64 → 128)
- Text branch: 128 → 256 dimensions (increased from 64 → 128)
- Simple concatenation fusion (no attention mechanism)
- Combined features: 512 dimensions (256 audio + 256 text)
- 1 dense hidden layer (128 units, increased from 64) after fusion
- Higher dropout (0.5) and reduced learning rate (0.0005)
- Output: 6 classes (CREMA-D) or 7 classes (MELD)

## Dependencies
Can be found in requirements.txt

## Output Files

After training, the following files are saved in `results/` (CREMA-D) or `results_meld/` (MELD), and figures in `figures/` or `figures_meld/`:

- `baseline_cnn.h5` - Trained baseline model
- `multimodal_cnn.h5` - Trained multimodal model
- `audio_scaler.pkl` - Audio feature scaler
- `text_scaler.pkl` - Text feature scaler
- `test_data.npz` - Test set for evaluation
- `confusion_matrices_comparison.png` - Side-by-side confusion matrices
- `f1_score_comparison.png` - Per-emotion F1 scores comparison
- `roc_curves_comparison.png` - ROC curves for all classes
- `evaluation_report.txt` - Detailed text report

## License
This project has a MIT license and is for academic purposes.