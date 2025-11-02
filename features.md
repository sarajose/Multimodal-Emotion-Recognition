# Features Explained

Quick reference for the audio and text features extracted from CREMA-D dataset.

## Audio Features

### 1. MFCCs
**What:** Voice texture/timbre - the "fingerprint" of how the voice sounds  
20 coefficients × 2 (mean + std) = 40 features  
**Emotion:** Different emotions have distinct voice qualities

### 2. Spectral Centroid
**What:** Brightness of the sound (weighted mean of frequencies)  
**Emotion:** Happy = bright/sharp, Sad = dull/dark

### 3. Spectral Rolloff
**What:** Frequency below which 85% of energy is concentrated  
**Emotion:** High-energy emotions (angry, happy) have higher rolloff

### 4. Spectral Bandwidth
**What:** Spread of frequencies (how wide the frequency range is)  
**Emotion:** Distressed emotions (fear, anger) have wider bandwidth

### 5. Zero Crossing Rate (ZCR)
**What:** How often the signal changes from positive to negative  
**Emotion:** Indicates speech clarity and intensity

### 6. Chroma
**What:** Musical pitch content (mean + std)  
**Emotion:** Different emotions use different pitch ranges

### 7. RMS Energy
**What:** Overall loudness/volume of the audio  
**Emotion:** Angry/happy = loud, Sad/fear = quiet

---

## Text Features

### 1. Sentiment Polarity 
**What:** How positive or negative the sentence is (-1 to +1)  
**Emotion:** Negative sentences = sad/angry, Positive = happy

### 2. Sentiment Subjectivity
**What:** How subjective vs objective the sentence is (0 to 1)  
**Emotion:** More subjective = emotional, Less = neutral

### 3. Word Count
**What:** Number of words in the sentence  
**Emotion:** May indicate complexity or urgency

### 4. Character Count
**What:** Total number of characters in the sentence  
**Emotion:** Similar to word count

### 5. Average Word Length
**What:** Mean length of words in the sentence  
**Emotion:** May correlate with formality or emotion type

---

## Summary
| Feature Type | Count | What It Measures |
|--------------|-------|------------------|
| **AUDIO FEATURES** | **47** | |
| MFCCs | 40 | Voice texture |
| Spectral Centroid | 1 | Brightness |
| Spectral Rolloff | 1 | Energy distribution |
| Spectral Bandwidth | 1 | Frequency spread |
| Zero Crossing Rate | 1 | Signal changes |
| Chroma | 2 | Pitch content |
| RMS Energy | 1 | Loudness |
| **TEXT FEATURES** | **5** | |
| Sentiment Polarity | 1 | Positive/negative |
| Sentiment Subjectivity | 1 | Subjective/objective |
| Word Count | 1 | Number of words |
| Character Count | 1 | Number of characters |
| Average Word Length | 1 | Mean word length |
