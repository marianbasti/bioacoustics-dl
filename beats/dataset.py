import logging
import torch
import torchaudio
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Union, Tuple, Optional
from torch.utils.data import Dataset
from functools import lru_cache
from concurrent.futures import ProcessPoolExecutor

logger = logging.getLogger(__name__)

class AudioDataset(Dataset):
    def __init__(
        self,
        root_dir: Union[str, Path],
        sample_rate: int = 16000,
        segment_duration: int = 10,
        overlap: float = 0.5,
        max_segments_per_file: Optional[int] = None,
        random_segments: bool = True,
        max_samples: Optional[int] = None,
        positive_dir: Optional[Union[str, Path]] = None,  # Directory with target sounds
        labeled_dir: Optional[Union[str, Path]] = None  # Directory with labeled sounds + CSV
    ) -> None:
        """
        Args:
            root_dir: Directory containing audio files
            sample_rate: Target sample rate in Hz
            segment_duration: Duration of each segment in seconds
            overlap: Overlap between segments (0.0 to 1.0)
            max_segments_per_file: Maximum segments to extract per file (None for all)
            random_segments: Whether to randomly select segments when max_segments_per_file is set
            max_samples: Maximum number of audio files to use (None for all)
            positive_dir: Directory containing positive examples for contrastive learning
            labeled_dir: Directory containing labeled examples and labels.csv
        """
        self.sample_rate = sample_rate
        self.segment_duration = segment_duration
        self.samples_per_segment = segment_duration * sample_rate
        self.overlap = min(max(0.0, overlap), 0.99)  # Clamp between 0 and 0.99
        self.hop_length = int(self.samples_per_segment * (1 - self.overlap))
        self.max_segments_per_file = max_segments_per_file
        self.random_segments = random_segments
        
        logger.info(f"Initializing AudioDataset with sample_rate={sample_rate}Hz, "
                   f"segment_duration={segment_duration}s, overlap={overlap:.1%}")
        
        # Get all audio files
        all_files: List[Path] = []
        for ext in ['*.wav', '*.mp3', '*.flac']:
            all_files.extend(Path(root_dir).rglob(ext))
        
        # Randomly sample files if max_samples is set
        if max_samples and max_samples < len(all_files):
            self.files = list(np.random.choice(all_files, max_samples, replace=False))
        else:
            self.files = all_files
            
        logger.info(f"Total available files: {len(all_files)}, Selected files: {len(self.files)}")
        
        # Pre-compute segment information for each file
        self.segments: List[Tuple[Path, int]] = []  # (file_path, segment_start)
        self._index_segments()
        
        logger.info(f"Total segments: {len(self.segments)}")
        
        self.positive_dir = Path(positive_dir) if positive_dir else None
        self.positive_files = []
        
        if self.positive_dir:
            for ext in ['*.wav', '*.mp3', '*.flac']:
                self.positive_files.extend(self.positive_dir.rglob(ext))
            logger.info(f"Found {len(self.positive_files)} positive examples")
            
        # Create segments for positive files
        self.positive_segments: List[Tuple[Path, int]] = []
        if self.positive_files:
            for file in self.positive_files:
                self.positive_segments.extend(self._analyze_file(file))
                
        self.labeled_dir = Path(labeled_dir) if labeled_dir else None
        self.labeled_files = []
        self.file_labels = {}
        self.unique_labels = set()
        
        if self.labeled_dir:
            # Load labels from CSV
            labels_file = self.labeled_dir / 'labels.csv'
            if labels_file.exists():
                df = pd.read_csv(labels_file)
                for _, row in df.iterrows():
                    filename = row['filename']
                    labels = eval(row['labels'])  # Convert string representation of list to actual list
                    filepath = self.labeled_dir / filename
                    if filepath.exists():
                        self.labeled_files.append(filepath)
                        self.file_labels[filepath] = labels
                        self.unique_labels.update(labels)
                
                logger.info(f"Found {len(self.labeled_files)} labeled files with {len(self.unique_labels)} unique labels")
            else:
                logger.error(f"Labels file not found: {labels_file}")
            
        # Create segments for labeled files
        self.labeled_segments: List[Tuple[Path, int]] = []
        if self.labeled_files:
            for file in self.labeled_files:
                self.labeled_segments.extend(self._analyze_file(file))
    
    def _analyze_file(self, file_path: Path) -> List[Tuple[Path, int]]:
        segments = []
        try:
            info = torchaudio.info(file_path)
            num_frames = info.num_frames
            num_segments = max(1, (num_frames - self.samples_per_segment) // self.hop_length + 1)
            
            if num_frames < self.samples_per_segment:
                segments.append((file_path, 0))
            else:
                # Create all possible segment start positions
                possible_starts = [i * self.hop_length for i in range(num_segments)]
                
                if self.max_segments_per_file and len(possible_starts) > self.max_segments_per_file:
                    if self.random_segments:
                        # Select random indices first
                        selected_indices = np.random.choice(
                            len(possible_starts),
                            self.max_segments_per_file,
                            replace=False
                        )
                        # Use indices to select start positions
                        selected_starts = [possible_starts[i] for i in selected_indices]
                        segments.extend((file_path, start) for start in selected_starts)
                    else:
                        # Take first n segments
                        segments.extend((file_path, start) 
                                     for start in possible_starts[:self.max_segments_per_file])
                else:
                    segments.extend((file_path, start) for start in possible_starts)
                        
        except Exception as e:
            logger.error(f"Error indexing file {file_path}: {str(e)}")
        
        return segments
    
    def _index_segments(self) -> None:
        with ProcessPoolExecutor() as executor:
            # Map the _analyze_file method to all files in parallel
            futures = [executor.submit(self._analyze_file, file_path) 
                      for file_path in self.files]
            
            # Collect results as they complete
            for future in futures:
                self.segments.extend(future.result())
    
    def __len__(self) -> int:
        return len(self.segments)
    
    @lru_cache(maxsize=100)
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        file_path, start_frame = self.segments[idx]
        
        try:
            waveform, sr = torchaudio.load(file_path, frame_offset=start_frame,
                                         num_frames=self.samples_per_segment)
            
            if sr != self.sample_rate:
                waveform = torchaudio.functional.resample(waveform, sr, self.sample_rate)
            
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            # Pad if needed (only for last segment or short files)
            if waveform.shape[1] < self.samples_per_segment:
                padding = self.samples_per_segment - waveform.shape[1]
                waveform = torch.nn.functional.pad(waveform, (0, padding))
            
            return waveform.squeeze(0), f"{file_path}:{start_frame}"
            
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {str(e)}")
            raise
    
    def get_positive_batch(self, batch_size: int) -> List[Tuple[torch.Tensor, str]]:
        """Get a batch of positive examples for contrastive learning"""
        if not self.positive_segments:
            return None
            
        indices = np.random.choice(len(self.positive_segments), 
                                 size=min(batch_size, len(self.positive_segments)),
                                 replace=False)
        return [self.__getitem__(idx) for idx in indices]
    
    def get_labeled_batch(self, batch_size: int) -> List[Tuple[torch.Tensor, str, List[str]]]:
        """Get a batch of labeled examples for contrastive learning"""
        if not self.labeled_segments:
            return None
            
        indices = np.random.choice(len(self.labeled_segments), 
                                 size=min(batch_size, len(self.labeled_segments)),
                                 replace=False)
        
        batch = []
        for idx in indices:
            file_path, start_frame = self.labeled_segments[idx]
            waveform, file_id = self.__getitem__(idx)
            labels = self.file_labels[file_path]
            batch.append((waveform, file_id, labels))
            
        return batch

    def get_all_labels(self):
        """Return set of all unique labels"""
        return self.unique_labels