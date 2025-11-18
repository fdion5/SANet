import torch
import torch.nn as nn
from torchvision.utils import save_image
from PIL import Image

from models.SANet import Net
from models.utils import input_transforms

#############################
EVAL                    = True
#############################

ENCODER_PATH            = "weights/vgg_normalised.pth"
TRANSFORMER_PATH        = "weights/transformer_iter_500000.pth"
DECODER_PATH            = "weights/decoder_iter_500000.pth"


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")



if EVAL:
    
    
    CONTENT_PATH        = "input/milo.jpg"
    STYLE_PATH          = "style/wave.jpg"
    NB_IT               = 1
    OUTPUT_PATH         = "output/try.jpg"

    model = Net(ENCODER_PATH, TRANSFORMER_PATH, DECODER_PATH)
    
    model.eval()

    model.to(DEVICE)
        
    content_image = input_transforms()
    content_image = content_image(Image.open(CONTENT_PATH))
    content_image = content_image.to(DEVICE).unsqueeze(0)

    style_image = input_transforms()
    style_image = style_image(Image.open(STYLE_PATH))
    style_image = style_image.to(DEVICE).unsqueeze(0)
    
    with torch.no_grad():
        
        for it in range(NB_IT):
            
            output = model(content_image, style_image)
            output.clamp(0, 255) #-> Clamping the pixels
            
        output = output.cpu()
        
        
        save_image(output, OUTPUT_PATH)
        
        
else:
    # params = filter(lambda x: x.requires_grad, model.parameters())

    # optimizer = torch.optim.Adam(params, lr=)
    pass