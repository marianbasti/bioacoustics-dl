import torch
import torchaudio
from pathlib import Path
from typing import List, Union
from torch.utils.data import Dataset

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
        self.sample_rate = sample_rate
        self.duration = duration
        self.samples = duration * sample_rate
        
        # Get all audio files recursively
        self.files: List[Path] = []
        for ext in ['*.wav', '*.mp3', '*.flac']:
            self.files.extend(list(Path(root_dir).rglob(ext)))
            
        # Cache for current audio file
        self.current_audio = None
        self.current_file_idx = -1
        self.current_position = 0
        
        # Calculate total number of clips
        self.total_clips = self._count_total_clips()
        
    def _count_total_clips(self) -> int:
        """Count total number of clips across all files."""
        total = 0
        for file_path in self.files:
            waveform, sr = torchaudio.load(file_path)
            if sr != self.sample_rate:
                waveform = torchaudio.functional.resample(waveform, sr, self.sample_rate)
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            total += max(1, waveform.shape[1] // self.samples)
        return total
        
    def _load_audio(self, file_idx: int) -> None:
        """Load and preprocess an audio file."""
        waveform, sr = torchaudio.load(self.files[file_idx])
        if sr != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, self.sample_rate)
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        self.current_audio = waveform
        self.current_file_idx = file_idx
        self.current_position = 0
    
    def __len__(self) -> int:
        """Return the total number of clips across all audio files."""
        return self.total_clips
    
    def __getitem__(self, idx: int) -> Union[torch.Tensor, str]:
        """Load and return a 10-second clip from the dataset.
        
        This method keeps track of the current audio file and position,
        returning consecutive non-overlapping 10-second segments.
        """
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
        else:
            # Handle audio shorter than desired duration
            waveform = torch.nn.functional.pad(self.current_audio, (0, self.samples - self.current_audio.shape[1]))
            self.current_position = 0
            self.current_file_idx = -1
            
        return waveform.squeeze(0), str(self.files[file_idx])