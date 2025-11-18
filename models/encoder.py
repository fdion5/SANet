import torch
import torch.nn as nn
from torchvision import models

from models.utils import VGG
from enum import IntEnum



class Layers(IntEnum):
    INPUT_FEATURES      = 0
    RELU_1_1            = 1
    RELU_2_1            = 2
    RELU_3_1            = 3
    RELU_4_1            = 4
    RELU_5_1            = 5
    END                 = 6




class Encoder(nn.Module):

    """
    VGG-19 encoder for style and content features 
    
    """
    
    def __init__(self, path:str = None):
        super().__init__()
        """
        [3]
        """
        
        vgg = VGG
        
        try:
            vgg.load_state_dict(torch.load(path))
        except FileNotFoundError as e:
            print("{0} is not a valid path for the encoder".format(path))
            raise
    
        self.relu_1_1 = nn.Sequential(*list(vgg.children())[:4])  
        self.relu_2_1 = nn.Sequential(*list(vgg.children())[4:11])  
        self.relu_3_1 = nn.Sequential(*list(vgg.children())[11:18])  
        self.relu_4_1 = nn.Sequential(*list(vgg.children())[18:31])  
        self.relu_5_1 = nn.Sequential(*list(vgg.children())[31:44])  


        """
        [2] [5]
        """
        #Freeze the layers. No training on them.
        for _,param in self.named_parameters():
            param.require_grad = False
        
        self.__layers = [self.relu_1_1, self.relu_2_1,self.relu_3_1, self.relu_4_1, self.relu_5_1]
        
        
    def forward(self, x:torch.Tensor) -> list[torch.Tensor]:
        """Compute the inference for the five layers

        Args:
            x (torch.Tensor): image
        """
        
        out = [x]
        
        for layer_index in range(len(self.__layers)):
            out.append(self.__layers[layer_index](out[layer_index]))
            
        return out
    

        
        
