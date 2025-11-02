# Emotion Recognition: Baseline vs Multimodal CNN

A comparison of **audio-only** vs **multimodal (audio+text)** CNN architectures for emotion recognition on the CREMA-D dataset.

## Overview

This project compares two deep learning (CNN) approaches for speech emotion recognition:
1. **Baseline CNN**: Audio features only
2. **Multimodal CNN**: Audio + Text features

## Dataset

**CREMA-D (Crowd-sourced Emotional Multimodal Actors Dataset)**
- 7,442 audio clips (.wav files)
- 6 emotions: Angry, Disgust, Fear, Happy, Neutral, Sad
- 91 actors (48 male, 43 female)

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

```bash
# Train models only
python train.py --data-path data

# Evaluate trained models and interactive analysis
jupyter notebook model_evaluation.ipynb
```
This will:
1. Extract audio and text features from CREMA-D dataset
2. Train both Baseline and Multimodal CNN models
3. Evaluate and generate visualizations

## Model Architectures

### Baseline CNN (audio only)
```
Input (47, 1) 
→ Conv1D(64, kernel=3) + MaxPool(2) + Dropout(0.3)
→ Conv1D(128, kernel=3) + GlobalAvgPool + Dropout(0.4)
→ Dense(64, relu) + Dropout(0.4)
→ Dense(6, softmax) [Output]
```
**Architecture Details:**
- 2 convolutional layers (64 → 128 filters) to extract audio patterns
- MaxPooling and Dropout for regularization
- 1 dense hidden layer (64 units)
- ~50,000 parameters

### Multimodal CNN (audio + text)
```
Audio Branch: 
  Input (47, 1) 
  → Conv1D(64, kernel=3) + MaxPool(2)
  → Conv1D(128, kernel=3) + GlobalAvgPool
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