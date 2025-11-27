"""
Data loading utilities for GTZAN music genre dataset.
"""

from pathlib import Path
from typing import List, Tuple, Dict
import numpy as np
from sklearn.model_selection import train_test_split


class GTZANLoader:
    """Load and manage GTZAN music genre dataset."""
    
    def __init__(self, data_path: str):
        """
        Initialize the data loader.
        
        Args:
            data_path: Path to the root data directory
        """
        self.data_path = Path(data_path)
        self.audio_path = self.data_path / 'raw' / 'genres_original'
        self.genres = None
        self.file_list = []
        
    def get_genres(self) -> List[str]:
        """
        Get list of available genres.
        
        Returns:
            List of genre names
        """
        if self.genres is None:
            self.genres = sorted([d.name for d in self.audio_path.iterdir() 
                                 if d.is_dir()])
        return self.genres
    
    def get_audio_files(self, genre: str = None) -> List[Tuple[str, str]]:
        """
        Get list of audio files with their genre labels.
        
        Args:
            genre: Specific genre to filter by (optional)
            
        Returns:
            List of (file_path, genre_label) tuples
        """
        files = []
        genres_to_process = [genre] if genre else self.get_genres()
        
        for g in genres_to_process:
            genre_path = self.audio_path / g
            if genre_path.exists():
                audio_files = sorted(genre_path.glob('*.wav'))
                files.extend([(str(f), g) for f in audio_files])
        
        return files
    
    def create_splits(
        self, 
        test_size: float = 0.15,
        val_size: float = 0.15,
        random_state: int = 42
    ) -> Tuple[List, List, List]:
        """
        Create train/validation/test splits.
        
        Args:
            test_size: Proportion of data for testing
            val_size: Proportion of remaining data for validation
            random_state: Random seed for reproducibility
            
        Returns:
            Tuple of (train_files, val_files, test_files)
        """
        files = self.get_audio_files()
        file_paths = [f[0] for f in files]
        labels = [f[1] for f in files]
        
        # First split: train+val vs test
        train_val_files, test_files, train_val_labels, test_labels = train_test_split(
            file_paths, labels, 
            test_size=test_size, 
            stratify=labels,
            random_state=random_state
        )
        
        # Second split: train vs val
        val_ratio = val_size / (1 - test_size)
        train_files, val_files, train_labels, val_labels = train_test_split(
            train_val_files, train_val_labels,
            test_size=val_ratio,
            stratify=train_val_labels,
            random_state=random_state
        )
        
        # Combine back into (path, label) tuples
        train = list(zip(train_files, train_labels))
        val = list(zip(val_files, val_labels))
        test = list(zip(test_files, test_labels))
        
        return train, val, test
    
    def get_genre_to_id_mapping(self) -> Dict[str, int]:
        """
        Get mapping from genre names to integer IDs.
        
        Returns:
            Dictionary mapping genre -> id
        """
        genres = self.get_genres()
        return {genre: idx for idx, genre in enumerate(genres)}
    
    def get_id_to_genre_mapping(self) -> Dict[int, str]:
        """
        Get mapping from integer IDs to genre names.
        
        Returns:
            Dictionary mapping id -> genre
        """
        genres = self.get_genres()
        return {idx: genre for idx, genre in enumerate(genres)}
