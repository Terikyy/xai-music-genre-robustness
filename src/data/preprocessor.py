"""
Preprocessing utilities for audio data and spectrogram generation.
"""

import numpy as np
import librosa
from pathlib import Path
from typing import Tuple, Optional
from tqdm import tqdm


class SpectrogramPreprocessor:
    """Generate and preprocess mel spectrograms from audio files."""
    
    def __init__(
        self,
        sr: int = 22050,
        n_mels: int = 128,
        n_fft: int = 2048,
        hop_length: int = 512,
        duration: float = 30.0
    ):
        """
        Initialize the preprocessor.
        
        Args:
            sr: Target sampling rate
            n_mels: Number of mel bands
            n_fft: FFT window size
            hop_length: Number of samples between frames
            duration: Audio duration in seconds
        """
        self.sr = sr
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.duration = duration
    
    def load_audio(self, file_path: str) -> np.ndarray:
        """
        Load audio file with librosa.
        
        Args:
            file_path: Path to audio file
            
        Returns:
            Audio time series
        """
        y, _ = librosa.load(file_path, sr=self.sr, duration=self.duration)
        return y
    
    def audio_to_spectrogram(self, y: np.ndarray) -> np.ndarray:
        """
        Convert audio to mel spectrogram.
        
        Args:
            y: Audio time series
            
        Returns:
            Mel spectrogram in dB scale
        """
        # Compute mel spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=y,
            sr=self.sr,
            n_mels=self.n_mels,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )
        
        # Convert to dB scale
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        return mel_spec_db
    
    def normalize_spectrogram(self, spec: np.ndarray) -> np.ndarray:
        """
        Normalize spectrogram to [0, 1] range.
        
        Args:
            spec: Mel spectrogram in dB
            
        Returns:
            Normalized spectrogram
        """
        # Min-max normalization
        spec_min = spec.min()
        spec_max = spec.max()
        
        if spec_max - spec_min != 0:
            spec_norm = (spec - spec_min) / (spec_max - spec_min)
        else:
            spec_norm = spec - spec_min
            
        return spec_norm
    
    def process_audio_file(
        self, 
        file_path: str,
        normalize: bool = True
    ) -> np.ndarray:
        """
        Complete preprocessing pipeline for a single audio file.
        
        Args:
            file_path: Path to audio file
            normalize: Whether to normalize to [0, 1]
            
        Returns:
            Preprocessed spectrogram
        """
        # Load audio
        y = self.load_audio(file_path)
        
        # Generate spectrogram
        spec = self.audio_to_spectrogram(y)
        
        # Normalize if requested
        if normalize:
            spec = self.normalize_spectrogram(spec)
        
        return spec
    
    def process_dataset(
        self,
        file_label_pairs: list,
        genre_to_id: dict,
        save_path: Optional[str] = None,
        normalize: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Process multiple audio files into spectrograms.
        
        Args:
            file_label_pairs: List of (file_path, genre_label) tuples
            genre_to_id: Dictionary mapping genre names to IDs
            save_path: Optional path to save processed data
            normalize: Whether to normalize spectrograms
            
        Returns:
            Tuple of (spectrograms, labels) arrays
        """
        spectrograms = []
        labels = []
        expected_shape = self.get_spectrogram_shape()
        
        print(f"Processing {len(file_label_pairs)} audio files...")
        
        for file_path, genre in tqdm(file_label_pairs):
            try:
                # Process audio file
                spec = self.process_audio_file(file_path, normalize=normalize)
                
                # Validate shape
                if spec.shape != expected_shape:
                    print(f"Warning: Skipping {file_path} - shape mismatch {spec.shape} vs {expected_shape}")
                    continue
                
                # Store results
                spectrograms.append(spec)
                labels.append(genre_to_id[genre])
                
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
                continue
        
        # Convert to numpy arrays
        X = np.array(spectrograms)
        y = np.array(labels)
        
        # Add channel dimension for CNN (samples, height, width, channels)
        X = np.expand_dims(X, axis=-1)
        
        print(f"Processed shape: X={X.shape}, y={y.shape}")
        
        # Save if path provided
        if save_path:
            save_path = Path(save_path)
            save_path.mkdir(parents=True, exist_ok=True)
            
            np.save(save_path / 'X.npy', X)
            np.save(save_path / 'y.npy', y)
            print(f"Saved to {save_path}")
        
        return X, y
    
    def get_spectrogram_shape(self) -> Tuple[int, int]:
        """
        Calculate expected spectrogram dimensions.
        
        Returns:
            Tuple of (n_mels, time_steps)
        """
        time_steps = int(np.ceil(self.duration * self.sr / self.hop_length))
        return (self.n_mels, time_steps)
