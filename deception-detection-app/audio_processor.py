"""
Audio processing module for deception detection
Handles speech pattern analysis and feature extraction
"""

import numpy as np
import librosa
import speech_recognition as sr

class AudioAnalyzer:
    """Analyze audio for deception cues"""
    
    def __init__(self, audio_path):
        self.audio_path = audio_path
        self.y, self.sr = None, None
        self.duration = 0
    
    def load_audio(self):
        """Load audio file"""
        try:
            self.y, self.sr = librosa.load(self.audio_path, sr=None, duration=60)
            self.duration = len(self.y) / self.sr if self.sr else 0
            return self.y, self.sr
        except Exception as e:
            print(f"Error loading audio: {e}")
            return None, None
    
    def extract_all_features(self):
        """Extract all audio features"""
        if self.y is None:
            return None
        
        features = {}
        
        try:
            # Tempo
            tempo, _ = librosa.beat.beat_track(y=self.y, sr=self.sr)
            features['speech_tempo'] = float(tempo) if isinstance(tempo, (int, float)) else 120.0
            
            # Spectral centroid
            spectral_centroids = librosa.feature.spectral_centroid(y=self.y, sr=self.sr)[0]
            features['avg_spectral_centroid'] = float(np.mean(spectral_centroids))
            features['std_spectral_centroid'] = float(np.std(spectral_centroids))
            
            # RMS energy
            rms = librosa.feature.rms(y=self.y)[0]
            features['avg_energy'] = float(np.mean(rms))
            features['std_energy'] = float(np.std(rms))
            features['energy_range'] = float(np.max(rms) - np.min(rms))
            
            # Pitch
            pitches, magnitudes = librosa.piptrack(y=self.y, sr=self.sr)
            pitch_values = []
            for i in range(pitches.shape[1]):
                index = magnitudes[:, i].argmax()
                pitch = pitches[index, i]
                if pitch > 0:
                    pitch_values.append(pitch)
            
            if pitch_values:
                features['avg_pitch'] = float(np.mean(pitch_values))
                features['std_pitch'] = float(np.std(pitch_values))
                features['pitch_range'] = float(np.max(pitch_values) - np.min(pitch_values))
            else:
                features['avg_pitch'] = 150.0
                features['std_pitch'] = 20.0
                features['pitch_range'] = 80.0
            
            # Zero crossing rate
            zcr = librosa.feature.zero_crossing_rate(self.y)[0]
            features['avg_zcr'] = float(np.mean(zcr))
            features['speech_activity'] = float(np.mean(zcr > 0.01))
            
            # MFCCs
            mfccs = librosa.feature.mfcc(y=self.y, sr=self.sr, n_mfcc=13)
            for i in range(13):
                features[f'mfcc_{i}_mean'] = float(np.mean(mfccs[i]))
                features[f'mfcc_{i}_std'] = float(np.std(mfccs[i]))
            
            # Spectral rolloff
            rolloff = librosa.feature.spectral_rolloff(y=self.y, sr=self.sr)[0]
            features['avg_rolloff'] = float(np.mean(rolloff))
            
            # Spectral bandwidth
            bandwidth = librosa.feature.spectral_bandwidth(y=self.y, sr=self.sr)[0]
            features['avg_bandwidth'] = float(np.mean(bandwidth))
            
        except Exception as e:
            print(f"Audio feature extraction error: {e}")
            # Return default values
            features = self._get_default_features()
        
        return features
    
    def _get_default_features(self):
        """Return default feature values"""
        return {
            'speech_tempo': 120.0,
            'avg_energy': 0.5,
            'std_energy': 0.05,
            'avg_pitch': 150.0,
            'std_pitch': 20.0,
            'speech_activity': 0.5,
            'energy_range': 0.1,
            'avg_zcr': 0.05
        }
    
    def transcribe_audio(self):
        """Transcribe audio to text"""
        try:
            recognizer = sr.Recognizer()
            with sr.AudioFile(self.audio_path) as source:
                recognizer.adjust_for_ambient_noise(source, duration=1)
                audio = recognizer.record(source)
                text = recognizer.recognize_google(audio)
                return text
        except Exception as e:
            print(f"Transcription error: {e}")
            return ""

def calculate_audio_deception_score(audio_features):
    """Calculate deception probability from audio features"""
    if not audio_features:
        return 0
    
    score = 0
    
    # Fast speech rate (anxiety indicator)
    if audio_features.get('speech_tempo', 120) > 160:
        score += 20
    elif audio_features.get('speech_tempo', 120) < 100:
        score += 15
    
    # High pitch variation (stress indicator)
    if audio_features.get('std_pitch', 0) > 30:
        score += 20
    elif audio_features.get('std_pitch', 0) > 20:
        score += 10
    
    # High energy variation (emotional arousal)
    if audio_features.get('std_energy', 0) > 0.08:
        score += 15
    elif audio_features.get('std_energy', 0) > 0.05:
        score += 8
    
    # Large energy range (inconsistent)
    if audio_features.get('energy_range', 0) > 0.2:
        score += 15
    elif audio_features.get('energy_range', 0) > 0.15:
        score += 8
    
    # Low speech activity (hesitation)
    if audio_features.get('speech_activity', 0) < 0.3:
        score += 15
    elif audio_features.get('speech_activity', 0) < 0.4:
        score += 8
    
    return min(100, score)