import torch
import torch.nn as nn
from torchvision.utils import save_image
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn

from PIL import Image
from tqdm import tqdm
import fiftyone.zoo as foz
from datasets import load_dataset
import io

from models.SANet import Net
from evaluation import run_lpips, run_fid
from models.utils import input_transforms, train_transform, \
                        FlatFolderDataset, InfiniteSamplerWrapper, learning_rate, \
                        save_model




##################################################################
EVAL                    = False
RATE                    = True
TRAIN                   = not EVAL and not RATE
##################################################################

########################## - WEIGHTS - ##########################

ENCODER_PATH            = "weights/vgg_normalised.pth"
#TRANSFORMER_PATH        = "weights/tr_transformer_10000.pth"
#DECODER_PATH            = "weights/tr_decoder_10000.pth"
TRANSFORMER_PATH        = "weights/transformer_iter_500000.pth"
DECODER_PATH            = "weights/decoder_iter_500000.pth"
OPTIMIZER_PATH          = None

##################################################################

############################ - RATE - ############################

OUTPUT_FOLDER               = "evaluation/eval_output/"
BUILD_OUTPUT                = False
INPUT_FOLDER_DICT           = {
    "monet": "evaluation/eval_input/monet.parquet",
    "cezanne": "evaluation/eval_input/cezanne.parquet",
    "ukiyoe": "evaluation/eval_input/ukiyoe.parquet",
    "vangogh": "evaluation/eval_input/vangogh.parquet",
}

##################################################################

############################ - EVAL - ############################

CONTENT_IMG_PATH        = "input/chicago.jpg"
STYLE_IMG_PATH          = "style/wave.jpg"
NB_IT                   = 1
OUTPUT_PATH             = "output/try.jpg"

##################################################################

########################## - TRAINING - ##########################

STYLE_PATH              = "train_1" # Needs to be already downloaded https://www.kaggle.com/competitions/painter-by-numbers . 
CONTENT_PATH            = "/train/data"
CONTENT_DATASET         = "coco-2017"
NUM_OF_COCO_SAMPLE      = 250
DEVICE                  = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LEARNING_RATE           = 1e-4
MAX_ITER                = 200
BATCH_SIZE              = 5

CONTENT_WEIGHT          = 1.0
STYLE_WEIGHT            = 3.0
L1_WEIGHT               = 50
L2_WEIGHT               = 1

cudnn.benchmark         = True

BREAK                   = False
##################################################################

##################################################################
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
elif RATE:
    # Initialize the model
    model = Net(ENCODER_PATH, TRANSFORMER_PATH, DECODER_PATH)
    model.eval()
    model.to(DEVICE)   

    if BUILD_OUTPUT:
        # Build rating dataset
        for style in INPUT_FOLDER_DICT.keys():
            print(f"Processing style: {style}")
            dataset = load_dataset("parquet", data_files={'test': INPUT_FOLDER_DICT[style]}, split='test')
            for i, data in enumerate(dataset):
                style_bytes = data['imageA']['bytes']
                style_image = input_transforms()
                style_image = style_image(Image.open(io.BytesIO(style_bytes)))
                style_image = style_image.to(DEVICE).unsqueeze(0)

                content_bytes = data['imageB']['bytes']
                content_image = input_transforms()
                content_image = content_image(Image.open(io.BytesIO(content_bytes)))
                content_image = content_image.to(DEVICE).unsqueeze(0)
                
                with torch.no_grad():
                    output = model(content_image, style_image)
                    output.clamp(0, 255)
                    output = output.cpu()
                    print(f"Saving image: {OUTPUT_FOLDER + f"{style}/" + f"{i}.jpg"}")
                    save_image(output, OUTPUT_FOLDER + f"{style}/" + f"{i}.jpg")
                    # raise Exception("Stop after one image for testing.")

    # Run LPIPS evaluation
    # run_lpips(INPUT_FOLDER_DICT, OUTPUT_FOLDER)

    # Run FID evaluation
    run_fid(INPUT_FOLDER_DICT, OUTPUT_FOLDER)
        
##################################################################


##################################################################
elif TRAIN:
    
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
        
        
    content_tf = train_transform()
    style_tf = train_transform()

    content_dataset = FlatFolderDataset(content_path, content_tf)
    style_dataset = FlatFolderDataset(STYLE_PATH, style_tf)

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
    
    

    params = filter(lambda x: x.requires_grad, model.parameters())
    
    optimizer = torch.optim.Adam(params, lr=LEARNING_RATE)
    if not OPTIMIZER_PATH is None:
        try:
            optimizer.load_state_dict(torch.load(OPTIMIZER_PATH))
        except:
            None

    

    
    
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

            print("[X] - Epoch {0}, Loss:{1}".format(i, loss.item()))
        
        except :
            save_model(model, optimizer)
            raise
        
        
        if BREAK:
            break
        
    save_model(model, optimizer)

    
##################################################################
