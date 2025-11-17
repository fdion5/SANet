import torch
import torch.nn as nn

from utils import SANet as SANet_model
from utils import Transform as Transformer_model

from utils import mean_variance_normalization as mean_variance_norm






class SANet(nn.Module):
    
    def __init__(self, pretrain:bool = True, model:nn.Module = None, nb_features:int = 512):
        super().__init__()
        
        """
        [4]
        """
        self.conv_f = nn.Conv2d(in_channels=nb_features, out_channels=nb_features, kernel_size=1)
        self.conv_g = nn.Conv2d(in_channels=nb_features, out_channels=nb_features, kernel_size=1)
        self.conv_h = nn.Conv2d(in_channels=nb_features, out_channels=nb_features, kernel_size=1)
        
        self.softmax = nn.Softmax(dim=-1)
        
        # Added the output convolution here so it matches the pre defined model. 
        # Not supposed to be in this module
        self.conv_out = nn.Conv2d(in_channels=nb_features, out_channels=nb_features, kernel_size=1)
        
        
        if pretrain:
            
            if model is None:
                raise ValueError("No model available for pretrained SANet")
            self.conv_f = model.f
            self.conv_g= model.g
            self.conv_h = model.h
            
            self.conv_out = model.out_conv
            
            
        
    def forward(self, content:torch.Tensor, style:torch.Tensor) -> torch.Tensor:
        
        """
        [4] [10]
        """
        
        F = self.conv_f(mean_variance_norm(content))
        G = self.conv_g(mean_variance_norm(style))
        H = self.conv_h(style)
        
        n, c, h, w = F.size()
        F = F.view(n,-1,h*w).permute(0, 2, 1) # F transpose for dot product
        
        n, c, h, w = G.size()
        G = G.view(n,-1,h*w) 
        
        S = self.softmax(torch.bmm(F,G)) #dot product + softmax
        S = S.permute(0, 2, 1)# S transpose for dot product
        
        
        n, c, h, w = H.size()
        H = H.view(n,-1,h*w) 
        
        out = torch.bmm(H,S) # dot product
        out = out.view(content.size())
        
        out = self.conv_out(out)
        
        return out
        
        
        
        
class Transformer(nn.Module):
    
    def __init__(self,path:str, pretrain:bool = True, nb_features:int = 512) -> None:
        super().__init__()
        
        """
        [6]
        """
        
        self.sanet_4_1 = SANet(pretrain = False, model= None, nb_features = nb_features)
        self.sanet_5_1 = SANet(pretrain = False, model= None, nb_features = nb_features)
        self.upsamp_5_1 = nn.Upsample(scale_factor=2, mode='nearest')
        
        self.pad = nn.ReflectionPad2d((1, 1, 1, 1)) # Needed to keep the same nb_features after the conv2D.
        self.conv_merge = nn.Conv2d(in_channels=nb_features, out_channels=nb_features, kernel_size=3)
        
        
        if pretrain:
            model = Transformer_model(nb_features)
            try:
                model.load_state_dict(torch.load(path))
            except FileNotFoundError:
                print("{0} is not a valid path for the transformer".format(path))
                raise
            
            self.sanet_4_1 = SANet(pretrain=True, model = model.sanet4_1)
            self.sanet_5_1 = SANet(pretrain=True, model = model.sanet5_1)
            self.conv_merge = model.merge_conv
                
                
                
    def forward(self, content_4_1:torch.Tensor, style_4_1:torch.Tensor, content_5_1:torch.Tensor, style_5_1:torch.Tensor) -> torch.Tensor:
        """
        [9]
        """
        
        feat_4_1 = self.sanet_4_1(content_4_1, style_4_1) + content_4_1 # Attention module
        feat5_1 = self.upsamp_5_1((self.sanet_5_1(content_5_1, style_5_1)) + content_5_1) # Attention module
        
        feat = feat5_1 + feat_4_1
        
        out = self.conv_merge(self.pad(feat)) # 3x3
        
        return out
        
                
