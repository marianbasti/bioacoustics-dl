import torch
import torchaudio
import logging
from pathlib import Path
from typing import List, Union
from torch.utils.data import Dataset

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AudioDataset(Dataset):
    """Audio dataset loader for processing audio files of various formats.
    
    This dataset handles loading, resampling, and preprocessing of audio files.
    It supports WAV, MP3, and FLAC formats, converting them to a unified format
    with consistent sampling rate and duration.
    
    Args:
        root_dir (Union[str, Path]): Directory containing audio files
        sample_rate (int, optional): Target sample rate in Hz. Defaults to 16000.
        duration (int, optional): Target duration in seconds. Defaults to 10.
    """
    
    def __init__(self, root_dir: Union[str, Path], sample_rate: int = 16000, duration: int = 10) -> None:
        logger.info(f"Initializing AudioDataset with sample_rate={sample_rate}Hz, duration={duration}s")
        self.sample_rate = sample_rate
        self.duration = duration
        self.samples = duration * sample_rate
        
        # Get all audio files recursively
        self.files: List[Path] = []
        for ext in ['*.wav', '*.mp3', '*.flac']:
            self.files.extend(list(Path(root_dir).rglob(ext)))
            
        logger.info(f"Found {len(self.files)} audio files")
        
        # Cache for current audio file
        self.current_audio = None
        self.current_file_idx = -1
        self.current_position = 0
        
        # Calculate total number of clips
        self.total_clips = self._count_total_clips()
        logger.info(f"Total number of {duration}-second clips: {self.total_clips}")
        
    def _count_total_clips(self) -> int:
        """Count total number of clips across all files."""
        total = 0
        for file_path in self.files:
            logger.debug(f"Counting clips in {file_path}")
            waveform, sr = torchaudio.load(file_path)
            if sr != self.sample_rate:
                waveform = torchaudio.functional.resample(waveform, sr, self.sample_rate)
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            clips_in_file = max(1, waveform.shape[1] // self.samples)
            total += clips_in_file
            logger.debug(f"File {file_path}: {clips_in_file} clips")
        return total
        
    def _load_audio(self, file_idx: int) -> None:
        """Load and preprocess an audio file."""
        file_path = self.files[file_idx]
        logger.debug(f"Loading audio file: {file_path}")
        waveform, sr = torchaudio.load(file_path)
        if sr != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, self.sample_rate)
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        self.current_audio = waveform
        self.current_file_idx = file_idx
        self.current_position = 0
        logger.debug(f"Loaded audio shape: {self.current_audio.shape}")
    
    def __len__(self) -> int:
        """Return the total number of clips across all audio files."""
        return self.total_clips
    
    def __getitem__(self, idx: int) -> Union[torch.Tensor, str]:
        logger.debug(f"Getting item {idx}/{self.total_clips}")
        # Find which file and position corresponds to this idx
        clips_counted = 0
        file_idx = 0
        
        while file_idx < len(self.files):
            if self.current_file_idx != file_idx:
                self._load_audio(file_idx)
            
            num_clips = max(1, self.current_audio.shape[1] // self.samples)
            if clips_counted + num_clips > idx:
                # This is the file we want
                break
            clips_counted += num_clips
            file_idx += 1
            
        # Get the clip from the current position
        if self.current_audio.shape[1] >= self.samples:
            waveform = self.current_audio[:, self.current_position:self.current_position + self.samples]
            self.current_position += self.samples
            
            # If we've reached the end of the file, reset position
            if self.current_position + self.samples > self.current_audio.shape[1]:
                self.current_position = 0
                self.current_file_idx = -1
            logger.debug(f"Extracted clip from {self.files[file_idx]}, position {self.current_position}")
        else:
            # Handle audio shorter than desired duration
            waveform = torch.nn.functional.pad(self.current_audio, (0, self.samples - self.current_audio.shape[1]))
            self.current_position = 0
            self.current_file_idx = -1
            logger.debug(f"Padded short audio from {self.files[file_idx]}")
            
        return waveform.squeeze(0), str(self.files[file_idx])