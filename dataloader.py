import torch
from torch.utils.data import Dataset
from pathlib import Path
import torchaudio



## creating the custom dataloader
class ESC50(Dataset):
    def __init__(self,path):
        ###getting the directory listing from the path 
        files = Path(path).glob('*.wav') 
        #tuple contaning the file name and the label separable
        self.items = [(f, int(f.name.split("-")[-1].replace(".wav", ""))) for f in files]
        self.length = len(self.items)  
              
    def __getitem__(self, index) -> torch.Tensor:
        filename, label = self.items[index]
        audio_tensor, sample_rate = torchaudio.load(filename)
        return audio_tensor, label
    
    def __len__(self) -> int:
        return self.length
    
