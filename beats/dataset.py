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
            
    def __len__(self) -> int:
        """Return the total number of audio files."""
        return len(self.files)
    
    def __getitem__(self, idx: int) -> Union[torch.Tensor, str]:
        """Load and preprocess an audio file.
        
        Args:
            idx (int): Index of the audio file to load
            
        Returns:
            Union[torch.Tensor, str]: Processed audio waveform of shape [samples]
                                      with sample_rate and duration as specified in __init__,
                                      and the filename as a string.
        """
        audio_path = self.files[idx]
        
        # Load and resample if needed
        waveform, sr = torchaudio.load(audio_path)
        if sr != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, self.sample_rate)
            
        # Convert to mono by averaging channels if stereo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
            
        # Handle duration by either trimming or padding
        if waveform.shape[1] > self.samples:
            # Randomly crop if too long
            start = torch.randint(0, waveform.shape[1] - self.samples, (1,))
            waveform = waveform[:, start:start + self.samples]
        else:
            # Zero-pad if too short
            padding = self.samples - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, padding))
            
        # Return waveform without squeezing to maintain channel dimension
        return waveform, str(audio_path)