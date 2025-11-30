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
        duration: float = 30.0,
        segment_duration: float = 3.0,
        use_segments: bool = True
    ):
        """
        Initialize the preprocessor.
        
        Args:
            sr: Target sampling rate
            n_mels: Number of mel bands
            n_fft: FFT window size
            hop_length: Number of samples between frames
            duration: Total audio duration in seconds
            segment_duration: Duration of each segment in seconds
            use_segments: Whether to split audio into segments
        """
        self.sr = sr
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.duration = duration
        self.segment_duration = segment_duration
        self.use_segments = use_segments
        
        # Calculate number of segments
        if self.use_segments:
            self.num_segments = int(self.duration / self.segment_duration)
        else:
            self.num_segments = 1
    
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
    
    def split_audio_into_segments(self, y: np.ndarray) -> list:
        """
        Split audio into non-overlapping segments.
        
        Args:
            y: Audio time series
            
        Returns:
            List of audio segments
        """
        segments = []
        segment_samples = int(self.segment_duration * self.sr)
        
        for i in range(self.num_segments):
            start_sample = i * segment_samples
            end_sample = start_sample + segment_samples
            segment = y[start_sample:end_sample]
            
            # Only add if segment has expected length
            if len(segment) == segment_samples:
                segments.append(segment)
        
        return segments
    
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
            Preprocessed spectrogram (or list of spectrograms if using segments)
        """
        # Load audio
        y = self.load_audio(file_path)
        
        if self.use_segments:
            # Split into segments and process each
            segments = self.split_audio_into_segments(y)
            spectrograms = []
            
            for segment in segments:
                spec = self.audio_to_spectrogram(segment)
                if normalize:
                    spec = self.normalize_spectrogram(spec)
                spectrograms.append(spec)
            
            return spectrograms
        else:
            # Process full audio
            spec = self.audio_to_spectrogram(y)
            if normalize:
                spec = self.normalize_spectrogram(spec)
            return spec
    
    def process_dataset(
        self,
        file_label_pairs: list,
        genre_to_id: dict,
        save_path: Optional[str] = None,
        normalize: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Process multiple audio files into spectrograms.
        
        Args:
            file_label_pairs: List of (file_path, genre_label) tuples
            genre_to_id: Dictionary mapping genre names to IDs
            save_path: Optional path to save processed data
            normalize: Whether to normalize spectrograms
            
        Returns:
            Tuple of (spectrograms, labels, track_ids) arrays
        """
        spectrograms = []
        labels = []
        track_ids = []
        expected_shape = self.get_spectrogram_shape()
        
        print(f"Processing {len(file_label_pairs)} audio files...")
        if self.use_segments:
            print(f"Each file will be split into {self.num_segments} segments of {self.segment_duration}s")
        
        track_id = 0
        
        for file_path, genre in tqdm(file_label_pairs):
            try:
                # Process audio file
                result = self.process_audio_file(file_path, normalize=normalize)
                
                if self.use_segments:
                    # Result is a list of spectrograms
                    for segment_spec in result:
                        # Validate shape
                        if segment_spec.shape != expected_shape:
                            print(f"Warning: Skipping segment from {file_path} - shape mismatch {segment_spec.shape} vs {expected_shape}")
                            continue
                        
                        # Store results - all segments from same track get same label and track_id
                        spectrograms.append(segment_spec)
                        labels.append(genre_to_id[genre])
                        track_ids.append(track_id)
                else:
                    # Result is a single spectrogram
                    if result.shape != expected_shape:
                        print(f"Warning: Skipping {file_path} - shape mismatch {result.shape} vs {expected_shape}")
                        continue
                    
                    spectrograms.append(result)
                    labels.append(genre_to_id[genre])
                    track_ids.append(track_id)
                
                # Increment track ID for next file
                track_id += 1
                
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
                continue
        
        # Convert to numpy arrays
        X = np.array(spectrograms)
        y = np.array(labels)
        track_ids = np.array(track_ids)
        
        # Add channel dimension for CNN (samples, height, width, channels)
        X = np.expand_dims(X, axis=-1)
        
        print(f"Processed shape: X={X.shape}, y={y.shape}, track_ids={track_ids.shape}")
        if self.use_segments:
            print(f"Total segments: {len(spectrograms)}, from {track_id} tracks")
        
        # Save if path provided
        if save_path:
            save_path = Path(save_path)
            save_path.mkdir(parents=True, exist_ok=True)
            
            np.save(save_path / 'X.npy', X)
            np.save(save_path / 'y.npy', y)
            np.save(save_path / 'track_ids.npy', track_ids)
            print(f"Saved to {save_path}")
        
        return X, y, track_ids
    
    def get_spectrogram_shape(self) -> Tuple[int, int]:
        """
        Calculate expected spectrogram dimensions.
        
        Returns:
            Tuple of (n_mels, time_steps)
        """
        if self.use_segments:
            # Calculate time steps for segment duration
            time_steps = int(np.ceil(self.segment_duration * self.sr / self.hop_length))
        else:
            # Calculate time steps for full duration
            time_steps = int(np.ceil(self.duration * self.sr / self.hop_length))
        
        return (self.n_mels, time_steps)
