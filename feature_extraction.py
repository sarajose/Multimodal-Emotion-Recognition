"""
Feature Extraction Module
Extracts audio and text features from CREMA-D dataset
"""

import os
import librosa
import numpy as np
import pandas as pd
from pathlib import Path
from textblob import TextBlob
import warnings
warnings.filterwarnings('ignore')

class FeatureExtractor:
    """Extract audio and text features from WAV files"""
    
    def __init__(self, data_path='data'):
        self.data_path = Path(data_path)
        
        # CREMA-D sentence mappings
        self.sentences = {
            'IEO': "It's eleven o'clock",
            'TIE': 'That is exactly what happened',
            'IOM': "I'm on my way to the meeting",
            'IWW': 'I wonder what this is about',
            'TAI': 'The airplane is almost full',
            'MTI': 'Maybe tomorrow it will be cold',
            'IWL': 'I would like a new alarm clock',
            'ITH': 'I think I have a doctor\'s appointment',
            'DFA': "Don't forget a jacket",
            'ITS': "I think I've seen this before",
            'TSI': 'The surface is slick',
            'WSI': 'We\'ll stop in a couple of minutes'
        }
        
        # Emotion mapping
        self.emotion_map = {
            'ANG': 0, 'DIS': 1, 'FEA': 2,
            'HAP': 3, 'NEU': 4, 'SAD': 5
        }
    
    def extract_audio_features(self, audio_path):
        """Extract 47 audio features from WAV file"""
        try:
            # Load audio
            y, sr = librosa.load(audio_path, sr=22050, duration=3.0)
            
            # Extract features
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
            chroma = librosa.feature.chroma_stft(y=y, sr=sr)
            spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
            spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
            zcr = librosa.feature.zero_crossing_rate(y)
            rms = librosa.feature.rms(y=y)
            
            # Combine all features into one vector
            feature_vector = np.concatenate([
                np.mean(mfccs, axis=1),      
                np.std(mfccs, axis=1),       
                [np.mean(spectral_centroid)], 
                [np.mean(spectral_rolloff)],  
                [np.mean(spectral_bandwidth)],
                [np.mean(zcr)],               
                [np.mean(chroma)],            
                [np.std(chroma)],             
                [np.mean(rms)]                
            ])
            
            return feature_vector
        except Exception as e:
            print(f"Error processing {audio_path}: {e}")
            return None
    
    def extract_text_features(self, sentence):
        """Extract text features from sentence"""
        try:
            blob = TextBlob(sentence)
            
            # Extract features
            polarity = blob.sentiment.polarity
            subjectivity = blob.sentiment.subjectivity
            word_count = len(sentence.split())
            char_count = len(sentence)
            avg_word_length = np.mean([len(word) for word in sentence.split()])
            
            return np.array([avg_word_length, char_count, polarity, subjectivity, word_count])
            
        except Exception as e:
            print(f"Error extracting text features: {e}")
            return np.zeros(5)
    
    def parse_filename(self, filename):
        """Parse CREMA-D filename: ActorID_Sentence_Emotion_Intensity.wav"""
        parts = filename.stem.split('_')
        if len(parts) >= 3:
            return {
                'actor_id': parts[0],
                'sentence_code': parts[1],
                'emotion': parts[2],
                'intensity': parts[3] if len(parts) > 3 else 'XX'
            }
        return None
    
    def extract_all_features(self, max_samples=None):
        """Extract features from all WAV files in the data folder"""
        
        # Get WAV files
        all_wav_files = list(self.data_path.glob('*.wav'))
        if max_samples:
            wav_files = all_wav_files[:max_samples]
        else:
            wav_files = all_wav_files
        print(f"Found {len(wav_files)} WAV files")
        
        # Process each file
        data = []
        for i, wav_file in enumerate(wav_files):
            # Show progress every 100 files
            if i % 100 == 0:
                print(f"  Processing {i}/{len(wav_files)}...")
            
            # Parse filename
            file_info = self.parse_filename(wav_file)
            if not file_info:
                continue  # Filename invalid
            
            # Get sentence
            sentence_text = self.sentences.get(file_info['sentence_code'], "")
            if not sentence_text:
                continue # not found
            
            # Extract audio features
            audio_features = self.extract_audio_features(str(wav_file))
            if audio_features is None:
                continue  # Extraction failed
            
            # Get emotion label
            emotion_code = file_info['emotion']
            if emotion_code not in self.emotion_map:
                continue  # Skip unknown
            
            # Store all information
            data.append({
                'filename': wav_file.name,
                'audio_features': audio_features,
                'text_features': self.extract_text_features(sentence_text),
                'emotion': self.emotion_map[emotion_code],
                'emotion_name': emotion_code
            })
        
        # Step 3: Return the complete dataset
        print(f"Extracted features from {len(data)} samples")
        return data


if __name__ == "__main__":
    # Quick test of this file
    extractor = FeatureExtractor('data')
    print("Testing feature extraction on 10 samples")
    data = extractor.extract_all_features(max_samples=10)
    
    if data:
        print(f"\nAudio feature dimension: {len(data[0]['audio_features'])}")
        print(f"Text feature dimension: {len(data[0]['text_features'])}")
        print(f"Emotions found: {set([d['emotion_name'] for d in data])}")
