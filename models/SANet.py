

import torch
import torch.nn as nn

from models.utils import mean_variance_normalization, compute_mean_std as mean_var_norm, compute_mean_std

from models.utils import input_transforms
### Encoder
from models.encoder import Encoder
from models.encoder import Layers as ENC_LAYERS



### Transformer
from models.transformer import Transformer


### Decoder
from models.decoder import Decoder






class Net(nn.Module):
    
    
    def __init__(self, enc_path:str, trans_path:str, decoder_path:str, pretrain: bool = True):
        super().__init__()
        
        self.encoder = Encoder(enc_path)
        self.transformer = Transformer(trans_path, pretrain)
        self.decoder = Decoder(decoder_path, pretrain)
        
        self.mse = nn.MSELoss() 
        
        
    def forward(self, content_feat, style_feat):
        
        list_content = self.encoder.forward(content_feat)
        list_style = self.encoder.forward(style_feat)
        
        trans_feat = self.transformer(list_content[ENC_LAYERS.RELU_4_1],list_style[ENC_LAYERS.RELU_4_1],
                                      list_content[ENC_LAYERS.RELU_5_1],list_style[ENC_LAYERS.RELU_5_1])
        
        
        decode_feat = self.decoder(trans_feat)
        
        if not self.training:
            return decode_feat
        """
        [11]
        """
        list_output = self.encoder(decode_feat)
        
        
        ### Content loss ###
        '''
        [12]
        '''
        
        L_c = self.__calc_feature_loss(list_output[ENC_LAYERS.RELU_4_1], list_content[ENC_LAYERS.RELU_4_1]) + \
              self.__calc_feature_loss(list_output[ENC_LAYERS.RELU_5_1], list_content[ENC_LAYERS.RELU_5_1])
              
        
        ### Style loss ###
        '''
        [13]
        '''
        
        L_s = 0
        for index in range(1, len(list_style)):
            L_s += self.__calc_var_mean_loss(list_output[index], list_style[index])
            
            
            
        ### Identity losses ###
        '''
        [14]
        '''
        I_cc = self.decoder(self.transformer(list_content[ENC_LAYERS.RELU_4_1],list_content[ENC_LAYERS.RELU_4_1],
                                      list_content[ENC_LAYERS.RELU_5_1],list_content[ENC_LAYERS.RELU_5_1]))
        
        I_ss = self.decoder(self.transformer(list_style[ENC_LAYERS.RELU_4_1],list_style[ENC_LAYERS.RELU_4_1],
                                      list_style[ENC_LAYERS.RELU_5_1],list_style[ENC_LAYERS.RELU_5_1]))
        
        L_1 = self.__calc_feature_loss(I_cc, list_content, norm=False) + self.__calc_feature_loss(I_ss, list_style, norm=False)
        

        F_cc = self.encoder(I_cc)
        F_ss = self.encoder(I_ss)
        
        L_2 = 0
        for index in range(len(1, len(list_style))):
            L_2 += self.__calc_feature_loss(F_cc[index], list_content[index]) + \
                self.__calc_feature_loss(F_ss[index], list_style[index])
                
                
        return L_c, L_s, L_1, L_2
    
        
    def __calc_feature_loss(self, features: torch.Tensor, targets:torch.Tensor, norm:bool = True):
        
        feature = mean_var_norm(features) if norm else features
        target = mean_var_norm(targets) if norm else targets
        loss = self.mse(feature, target)

        return loss
    
    
    def __calc_var_mean_loss(self, features: torch.Tensor, targets:torch.Tensor):
        
        std_target, mean_target = compute_mean_std(targets)
        std_features, mean_features = compute_mean_std(features)
        loss = self.mse(std_features, std_target) + self.mse(mean_features, mean_target)
        
        return loss
        

# vgg = models.vgg19(weights = "DEFAULT",progress = False).features
# for param in vgg.parameters():
#     param.requires_grad = False

# for layer_idx, layer in enumerate(vgg):
#     print(layer_idx, layer)






        
        
        
    
        
        