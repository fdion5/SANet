import torch
import torch.nn as nn

from models.utils import DECODER





class Decoder(nn.Module):
    
    def __init__(self, path:str, pretrain = True):
        super().__init__()
        
        self.model = DECODER
        
        
        if pretrain:
            try:
                self.model.load_state_dict(torch.load(path))
            except FileNotFoundError:
                print("{0} is not a valid path for the decoder".format(path))
                raise
        
        
        
    def forward(self, features):
        
        return self.model(features)