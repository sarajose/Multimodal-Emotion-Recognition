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
2. Extract the raw `.wav` files into the `data/` folder

### MELD (Multimodal EmotionLines Dataset)
- Multimodal emotion recognition dataset
- 7 emotions: anger, disgust, fear, joy, neutral, sadness, surprise

**Download:**
1. Download from [Kaggle - MELD Dataset](https://www.kaggle.com/datasets/zaber666/meld-dataset)
2. Extract the `MELD.Raw` folder into `MELD_raw_data/`

**Note:** Both data folders are not included in this repository due to size constraints.

## Project Structure

```
project/
├── feature_extraction.py      # Audio and text feature extraction
├── models.py                  # CNN architectures
├── train.py                   # Training pipeline
├── evaluate.py                # 
├── model_evaluation.ipynb     # Evaluation and visualization (Interactive analysis)
├── requirements.txt           # Dependencies
└── results/                   # Saved models and outputs
└── figures/   
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

# Evaluate trained models and interactive analysis
jupyter notebook model_evaluation.ipynb
```

This will:
1. Extract audio and text features from the complete dataset
2. Train both Baseline and Multimodal CNN models
3. Evaluate and generate visualizations

## Model Architectures

### Baseline CNN (audio only)
```
Input (47, 1) 
→ Conv1D(64) + MaxPool(2) + Dropout(0.3)
→ Conv1D(128) + GlobalAvgPool + Dropout(0.4)
→ Dense(64, relu) + Dropout(0.4)
→ Dense(6, softmax) [Output]
```
**Architecture Details:**
- 2 convolutional layers (64 → 128 filters) to extract audio patterns
- MaxPooling and Dropout for regularization
- 1 dense hidden layer (64 units)

### Multimodal CNN (audio + text)
```
Audio Branch: 
  Input (47, 1) 
  → Conv1D(64) + MaxPool(2)
  → Conv1D(128) + GlobalAvgPool
  → Dropout(0.3) → [128 features]

Text Branch:  
  Input (5,) 
  → Dense(64, relu)
  → Dense(128, relu) → [128 features]

Fusion: 
  Concatenate [audio + text] → (256,)
  → Dropout(0.4)
  → Dense(64, relu)
  → Dense(6, softmax) [Output]
```
**Architecture Details:**
- Audio and text branches process features separately
- Simple concatenation fusion (no attention mechanism)
- Combined features: 256 dimensions (128 audio + 128 text)
- 1 dense hidden layer (64 units) after fusion

## Dependencies
Can be found in requirements.txt

## Output Files

After training, the following files are saved in `results/` and in `figures/`:

- `baseline_cnn.h5` - Trained baseline model
- `multimodal_cnn.h5` - Trained multimodal model
- `audio_scaler.pkl` - Audio feature scaler
- `text_scaler.pkl` - Text feature scaler
- `test_data.npz` - Test set for evaluation
- `confusion_matrices.png` - Side-by-side confusion matrices
- `per_class_f1_comparison.png` - Per-emotion F1 scores
- `evaluation_report.txt` - Detailed text report

## License
This project has a MIT license and is for academic purposes.