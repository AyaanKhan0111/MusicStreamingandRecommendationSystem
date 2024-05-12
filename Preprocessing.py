import os
import librosa
import numpy as np
from pymongo import MongoClient
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Function to extract audio features
def extract_features(audio_path, sample_rate=22050, apply_scaling=False):
    try:
        audio, sr = librosa.load(audio_path, sr=sample_rate, duration=30)  # Load only first 30 seconds
    except Exception as e:
        print(f"Error loading {audio_path}: {e}")
        return None
    
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)  # Extract MFCCs
    spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)  # Spectral Centroid
    zero_crossing_rate = librosa.feature.zero_crossing_rate(audio)  # Zero-Crossing Rate

    # Calculate mean and standard deviation of features
    mfccs_mean = np.mean(mfccs, axis=1)
    mfccs_std = np.std(mfccs, axis=1)
    spectral_centroid_mean = np.mean(spectral_centroid)
    zero_crossing_rate_mean = np.mean(zero_crossing_rate)
    
    # Normalize or standardize features
    if apply_scaling:
        scaler = StandardScaler()  # or MinMaxScaler()
        mfccs_mean = scaler.fit_transform(mfccs_mean.reshape(-1, 1)).flatten()
        mfccs_std = scaler.transform(mfccs_std.reshape(-1, 1)).flatten()
        spectral_centroid_mean = scaler.transform(np.array([spectral_centroid_mean]).reshape(1, -1)).flatten()
        zero_crossing_rate_mean = scaler.transform(np.array([zero_crossing_rate_mean]).reshape(1, -1)).flatten()

    return {
        'mfccs_mean': mfccs_mean.tolist(),
        'mfccs_std': mfccs_std.tolist(),
        'spectral_centroid_mean': spectral_centroid_mean.tolist(),
        'zero_crossing_rate_mean': zero_crossing_rate_mean.tolist(),
        'filename': os.path.basename(audio_path)
    }

# Path to the dataset directory
dataset_path = '/media/hdoop/TOSHIBA EXT/fma_large_sample'

# Connect to MongoDB
client = MongoClient('localhost', 27017)
db = client['fma']
collection = db['audio_features_with_standardization_normalization']

# Process each audio file and store features in MongoDB
for root, dirs, files in os.walk(dataset_path):
    for filename in files:
        if filename.endswith('.mp3'):
            audio_path = os.path.join(root, filename)
            # Extract features with standardization
            features = extract_features(audio_path, apply_scaling=True)
            if features is not None:
                # Insert into MongoDB
                collection.insert_one(features)
                print(f"Processed: {filename}")

# Close MongoDB connection
client.close()
