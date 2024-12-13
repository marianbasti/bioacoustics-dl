
import torch
import torchaudio
from pathlib import Path
from torch.utils.data import Dataset

class AudioDataset(Dataset):
    def __init__(self, root_dir, sample_rate=16000, duration=10):
        self.sample_rate = sample_rate
        self.duration = duration
        self.samples = duration * sample_rate
        
        # Get all audio files recursively
        self.files = []
        for ext in ['*.wav', '*.mp3', '*.flac']:
            self.files.extend(list(Path(root_dir).rglob(ext)))
            
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        audio_path = self.files[idx]
        
        # Load and resample if needed
        waveform, sr = torchaudio.load(audio_path)
        if sr != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, self.sample_rate)
            
        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
            
        # Ensure correct shape
        if waveform.shape[1] > self.samples:
            start = torch.randint(0, waveform.shape[1] - self.samples, (1,))
            waveform = waveform[:, start:start + self.samples]
        else:
            # Pad if too short
            padding = self.samples - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, padding))
            
        return waveform.squeeze(0)