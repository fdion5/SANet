import os, io
import torch
import lpips
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from datasets import load_dataset

def evaluate_lpips(input_image, output_image, lpips_model, device):
    input_image = input_image.to(device)
    output_image = output_image.to(device)
    with torch.no_grad():
        lpips_value = lpips_model(input_image, output_image)
    return lpips_value.item()

def format_input_output_images(input_image, output_image, device):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    input_tensor = transform(input_image).unsqueeze(0).to(device)
    output_tensor = transform(output_image).unsqueeze(0).to(device)
    return input_tensor, output_tensor

def print_results(results):
    print("-" * 20)
    print("LPIPS Evaluation Results:")
    for style_name, res in results.items():
        print(f"Style: {style_name}, Average LPIPS: {res['average_lpips']:.4f} over {res['num_images']} images")
    print("-" * 20)

def run_lpips(input_folder_dict, output_folder_path):
    lpips_model = lpips.LPIPS(net='alex')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lpips_model.to(device)
    results = {}

    for style_name in input_folder_dict.keys():
        dataset = load_dataset("parquet", data_files={'test': input_folder_dict[style_name]}, split='test')
        output_dir = output_folder_path + f"{style_name}/"
        lpips_scores = []

        for i, data in tqdm(enumerate(dataset), desc=f"Processing {style_name}"):
            input_bytes = data['imageB']['bytes']
            input_image = Image.open(io.BytesIO(input_bytes)).convert("RGB")
            output_image = Image.open(output_dir + f"{i}.jpg").convert("RGB")

            input_tensor, output_tensor = format_input_output_images(input_image, output_image, device)
            lpips_score = evaluate_lpips(input_tensor, output_tensor, lpips_model, device)
            lpips_scores.append(lpips_score)

        average_lpips = sum(lpips_scores) / len(lpips_scores) if lpips_scores else float('inf')
        results[style_name] = {
            "average_lpips": average_lpips,
            "num_images": len(lpips_scores)
        }

    print_results(results)  