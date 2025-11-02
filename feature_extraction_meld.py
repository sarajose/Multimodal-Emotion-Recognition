"""
Feature Extraction Module for MELD Dataset
Extracts audio and text features from MELD multimodal dataset
"""

import os
import librosa
import numpy as np
import pandas as pd
from pathlib import Path
from textblob import TextBlob
import warnings
warnings.filterwarnings('ignore')

class MELDFeatureExtractor:
    """Extract audio and text features from MELD dataset"""
    
    def __init__(self, data_path='MELD_raw_data'):
        self.data_path = Path(data_path)
        
        # MELD emotion mapping (7 emotions)
        self.emotion_map = {
            'anger': 0,
            'disgust': 1,
            'fear': 2,
            'joy': 3,
            'neutral': 4,
            'sadness': 5,
            'surprise': 6
        }
    
    def extract_audio_features(self, audio_path):
        """Extract 47 audio features from video/audio file"""
        try:
            # Load audio from mp4 file
            y, sr = librosa.load(audio_path, sr=22050, duration=10.0)
            
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
        """Extract 5 text features from sentence"""
        try:
            if not sentence or not isinstance(sentence, str):
                return np.zeros(5)
            
            blob = TextBlob(sentence)
            
            # Extract features
            polarity = blob.sentiment.polarity
            subjectivity = blob.sentiment.subjectivity
            word_count = len(sentence.split())
            char_count = len(sentence)
            avg_word_length = np.mean([len(word) for word in sentence.split()]) if word_count > 0 else 0
            
            return np.array([avg_word_length, char_count, polarity, subjectivity, word_count])
            
        except Exception as e:
            print(f"Error extracting text features: {e}")
            return np.zeros(5)
    
    def extract_all_features(self, split='train', max_samples=None):
        """
        Extract features from MELD dataset for a specific split
        
        Args:
            split: 'train', 'dev', or 'test'
            max_samples: Maximum number of samples to process (for testing)
        
        Returns:
            DataFrame with features and labels
        """
        
        # Load CSV file 
        # Check both possible locations for CSV file
        csv_path = self.data_path / split / f'{split}_sent_emo.csv'
        if not csv_path.exists():
            # Try root directory
            csv_path = self.data_path / f'{split}_sent_emo.csv'
        
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
        
        print(f"Loading {split} data from {csv_path}")
        df = pd.read_csv(csv_path)
        
        if max_samples:
            df = df.head(max_samples)
        
        print(f"Processing {len(df)} utterances")
        
        # Process each utterance
        data = []
        successful = 0
        failed = 0
        
        for idx, row in df.iterrows():
            # Show progress every 100 samples
            if idx % 100 == 0:
                print(f"  Processing {idx}/{len(df)}(Success: {successful}, Failed: {failed})")
            
            # Get text and emotion
            utterance = str(row['Utterance'])
            emotion = str(row['Emotion']).lower()
            
            # Map emotion to label
            if emotion not in self.emotion_map:
                print(f"  Unknown emotion: {emotion} (skipped)")
                failed += 1
                continue
            
            # Construct audio file path
            dialogue_id = row['Dialogue_ID']
            utterance_id = row['Utterance_ID']
            audio_filename = f"dia{dialogue_id}_utt{utterance_id}.mp4"
            
            # Handle different folder names for each split
            if split == 'train':
                audio_folder = 'train_splits'
            elif split == 'dev':
                audio_folder = 'dev_splits_complete'
            elif split == 'test':
                audio_folder = 'output_repeated_splits_test'
            else:
                audio_folder = f'{split}_splits'
            
            audio_path = self.data_path / split / audio_folder / audio_filename
            
            # Check if audio file exists
            if not audio_path.exists():
                # print(f"  Audio file not found: {audio_filename}, skipping...")
                failed += 1
                continue
            
            # Extract audio features
            audio_features = self.extract_audio_features(str(audio_path))
            if audio_features is None:
                failed += 1
                continue
            
            # Extract text features
            text_features = self.extract_text_features(utterance)
            
            # Store all information
            data.append({
                'filename': audio_filename,
                'audio_features': audio_features,
                'text_features': text_features,
                'emotion': self.emotion_map[emotion],
                'emotion_name': emotion,
                'utterance': utterance,
                'dialogue_id': dialogue_id,
                'utterance_id': utterance_id
            })
            
            successful += 1
        
        print(f"\nCompleted successfully: {successful}, Failed: {failed}")
        print(f"Emotion distribution:")
        emotions_df = pd.DataFrame(data)
        if len(emotions_df) > 0:
            print(emotions_df['emotion_name'].value_counts())
        
        return pd.DataFrame(data)


def main():
    """Main function to extract features and save to file"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Extract features from MELD dataset')
    parser.add_argument('--data-path', type=str, default='MELD_raw_data',
                       help='Path to MELD dataset folder')
    parser.add_argument('--output-path', type=str, default='results_meld',
                       help='Path to save extracted features')
    parser.add_argument('--max-samples', type=int, default=None,
                       help='Maximum samples to process (for testing)')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_path, exist_ok=True)
    
    # Initialize extractor
    extractor = MELDFeatureExtractor(data_path=args.data_path)
    
    # Extract features for each split
    for split in ['train', 'dev', 'test']:
        print(f"Processing {split.upper()} split")
        
        try:
            # Extract features
            df = extractor.extract_all_features(split=split, max_samples=args.max_samples)
            
            if len(df) == 0:
                print(f"Warning: No data extracted for {split} split!")
                continue
            
            # Prepare data for saving
            X_audio = np.array([f for f in df['audio_features'].values])
            X_text = np.array([f for f in df['text_features'].values])
            y = np.array(df['emotion'].values)
            
            # Save to file
            output_file = os.path.join(args.output_path, f'{split}_features.npz')
            np.savez(
                output_file,
                X_audio=X_audio,
                X_text=X_text,
                y=y,
                filenames=df['filename'].values,
                emotion_names=df['emotion_name'].values,
                utterances=df['utterance'].values
            )
            
            print(f"\n{split.upper()} data saved to {output_file}")
            print(f"  Audio features shape: {X_audio.shape}")
            print(f"  Text features shape: {X_text.shape}")
            print(f"  Labels shape: {y.shape}")
            
        except Exception as e:
            print(f"Error processing {split} split: {e}")
            import traceback
            traceback.print_exc()

    print("Feature extraction complete")

if __name__ == '__main__':
    main()
