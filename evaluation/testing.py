import torch, io, sys, os
from torchvision.utils import save_image
from datasets import load_dataset
from PIL import Image

# Get the directory of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))
# Get the parent directory (which contains the 'models' folder)
parent_dir = os.path.dirname(current_dir)
# Add parent directory to path
sys.path.append(parent_dir)

from models.SANet import Net
from models.utils import input_transforms

# 1. Load the local parquet file
dataset = load_dataset("parquet", data_files={'test': 'evaluation/eval_input/cezanne.parquet'}, split='test')

img_dict = dataset[0] # Returns a PIL Image object
for key in img_dict.keys():
    print(key)
    for key_in_key in img_dict[key].keys():
        print("\t" + key_in_key)

idA = img_dict['imageA']['id']
imgA = img_dict['imageA']['bytes']
imageA = Image.open(io.BytesIO(imgA))
imageA.show()

imgB = img_dict['imageB']['bytes']
imageB = Image.open(io.BytesIO(imgB))
imageB.show()

ENCODER_PATH            = "weights/vgg_normalised.pth"
TRANSFORMER_PATH        = "weights/transformer_iter_500000.pth"
DECODER_PATH            = "weights/decoder_iter_500000.pth"
DEVICE                  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTPUT_PATH             = "output/try.jpg"

# Initialize the model
model = Net(ENCODER_PATH, TRANSFORMER_PATH, DECODER_PATH)
model.eval()
model.to(DEVICE)        

# Load images
content_image = input_transforms()
content_image = content_image(imageB)
content_image = content_image.to(DEVICE).unsqueeze(0)

style_image = input_transforms()
style_image = style_image(imageA)
style_image = style_image.to(DEVICE).unsqueeze(0)

with torch.no_grad():
    output = model(content_image, style_image)
    output.clamp(0, 255) #-> Clamping the pixels
    output = output.cpu()
    save_image(output, OUTPUT_PATH)