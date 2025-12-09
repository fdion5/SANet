import torch
import torch.nn as nn
from torchvision.utils import save_image
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn

from PIL import Image
from PIL import ImageFile
Image.MAX_IMAGE_PIXELS = None  # Disable DecompressionBombError
ImageFile.LOAD_TRUNCATED_IMAGES = True  # Disable OSError: image file is truncated

from tqdm import tqdm
#import fiftyone.zoo as foz

from models.SANet import Net
from models.utils import    input_transforms, train_transform, \
                            FlatFolderDataset, InfiniteSamplerWrapper, learning_rate, \
                            save_model




##################################################################
EVAL                    = True
##################################################################


########################## - WEIGHTS - ##########################

ENCODER_PATH            = "weights/vgg_normalised.pth"
TRANSFORMER_PATH        = "weights/tr_transformer.pth" # transformer_iter_500000 try1_transformer
DECODER_PATH            = "weights/tr_decoder.pth" #decoder_iter_500000 try1_decoder
OPTIMIZER_PATH          = None

##################################################################


############################ - EVAL - ############################

CONTENT_IMG_PATH        = "input/chicago.jpg"
STYLE_IMG_PATH          = "style/wave.jpg"
NB_IT                   = 1
OUTPUT_PATH             = "output/try.jpg"

##################################################################


########################## - TRAINING - ##########################

#STYLE_PATH              = "train_1"
#CONTENT_PATH            = "/train/data"
#CONTENT_DATASET         = "coco-2017"
#NUM_OF_COCO_SAMPLE      = 250
DEVICE                  = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NUM_OF_WORKER           = 12
LEARNING_RATE           = 1e-4
MAX_ITER                = 160000
BATCH_SIZE              = 5
INTERVAL                = 1000

CONTENT_WEIGHT          = 1.0
STYLE_WEIGHT            = 3.0
L1_WEIGHT               = 50
L2_WEIGHT               = 1

cudnn.benchmark         = True

BREAK                   = False
##################################################################



##################################################################
if __name__ == "__main__":
    if EVAL:

        model = Net(ENCODER_PATH, TRANSFORMER_PATH, DECODER_PATH)
        
        model.eval()

        model.to(DEVICE)
            
        content_image = input_transforms()
        content_image = content_image(Image.open(CONTENT_IMG_PATH))
        content_image = content_image.to(DEVICE).unsqueeze(0)

        style_image = input_transforms()
        style_image = style_image(Image.open(STYLE_IMG_PATH))
        style_image = style_image.to(DEVICE).unsqueeze(0)
        
        with torch.no_grad():
            
            for it in range(NB_IT):
                
                output = model(content_image, style_image)
                output.clamp(0, 255) #-> Clamping the pixels
                
            output = output.cpu()
            
            
            save_image(output, OUTPUT_PATH)
            
    ##################################################################


    ##################################################################
    else:

        """
            Load a Zoo dataset for the content from Zoo 
        """

        """
        content_path = None
        
        try:
            content_path = foz.find_zoo_dataset(CONTENT_DATASET) + CONTENT_PATH
        except ValueError:
            dataset = foz.load_zoo_dataset(
                CONTENT_DATASET,
                split="train",
                max_samples=NUM_OF_COCO_SAMPLE,
                shuffle=True,
            )
            content_path = foz.find_zoo_dataset(CONTENT_DATASET) + CONTENT_PATH
        """

        print(torch.version.cuda)
        print(torch.cuda.is_available())
        print(torch.cuda.get_device_name(0))
            
        content_tf = train_transform()
        style_tf = train_transform()

        #content_dataset = FlatFolderDataset(content_path, content_tf)
        #style_dataset = FlatFolderDataset(STYLE_PATH, style_tf)

        content_dataset = FlatFolderDataset("./train2014", content_tf)
        style_dataset = FlatFolderDataset("./train", style_tf)

        content_loader = DataLoader(
            content_dataset,
            batch_size=BATCH_SIZE,
            sampler=InfiniteSamplerWrapper(content_dataset),
        )

        style_loader = DataLoader(
            style_dataset,
            batch_size=BATCH_SIZE,
            sampler=InfiniteSamplerWrapper(style_dataset),
        )
        
        content_iter = iter(content_loader)
        style_iter = iter(style_loader)
        model = Net(ENCODER_PATH, TRANSFORMER_PATH, DECODER_PATH, pretrain=False)
        model.train()
        model.to(DEVICE)

        """
        params = filter(lambda x: x.requires_grad, model.parameters())
        optimizer = torch.optim.Adam(params, lr=LEARNING_RATE)
        if not OPTIMIZER_PATH is None:
            try:
                optimizer.load_state_dict(torch.load(OPTIMIZER_PATH))
            except:
                None
        """
        
        optimizer = torch.optim.Adam([
                    {'params': model.decoder.parameters()},
                    {'params': model.transformer.parameters()}], 
                    lr=LEARNING_RATE)

        
        for i in tqdm(range(MAX_ITER)):
            
            try:
                learning_rate(optimizer, LEARNING_RATE, 5e-5, i)
                
                content_images = next(content_iter).to(DEVICE)
                style_images = next(style_iter).to(DEVICE)
                
                
                optimizer.zero_grad()
                
                L_C, L_S, L_1, L_2 = model(content_images, style_images)
                
                L_C = L_C * CONTENT_WEIGHT
                L_S = L_S * STYLE_WEIGHT
                L_1 = L_1 * L1_WEIGHT
                L_2 = L_2 * L2_WEIGHT
                
                
                loss = L_C + L_S + L_1 + L_2
                
                
                loss.backward()
                optimizer.step()

                if (i + 1) % INTERVAL == 0 or (i + 1) == MAX_ITER:
                    print("[X] - Epoch {0}, Loss:{1}".format(i, loss.item()))
                    save_model(model, optimizer, (i + 1))
            
            except :
                save_model(model, optimizer, (i + 1))
                raise
            
            
            if BREAK:
                break
            
        save_model(model, optimizer, MAX_ITER)

    
##################################################################
