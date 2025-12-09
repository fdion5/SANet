from pytorch_fid.fid_score import calculate_fid_given_paths
from datasets import load_dataset
import io, os, shutil
import os
import shutil
from PIL import Image

def calculate_fid(input_folder, output_folder):
    paths = [input_folder, output_folder]

    # Calculate FID
    fid_value = calculate_fid_given_paths(
        paths, 
        batch_size=24, 
        device='mps', 
        dims=192,
        num_workers=0
    )

    return fid_value

def extract_parquet_to_folder(parquet_path, output_folder):

    dataset = load_dataset("parquet", data_files={'test': parquet_path}, split='test')

    for i, data in enumerate(dataset):
        style_bytes = data['imageA']['bytes']
        image = Image.open(io.BytesIO(style_bytes)).convert("RGB")
        image.save(os.path.join(output_folder, f"{i}.jpg"))

    return output_folder


def run_fid(input_folder_dict, output_folder):
    temp_folder = "temp/"

    if os.path.exists(temp_folder):
        shutil.rmtree(temp_folder)
    os.makedirs(temp_folder)

    results = {}
    for style, input_path in input_folder_dict.items():
        extract_parquet_to_folder(input_path, temp_folder)
        style_output_folder = os.path.join(output_folder, style)
        fid_value = calculate_fid(temp_folder, style_output_folder)
        results[style] = fid_value

        if os.path.exists(temp_folder):
            shutil.rmtree(temp_folder)
        os.makedirs(temp_folder)
    
    print("-" * 20)
    print("FID Evaluation Results:")
    for style_name, fid in results.items():
        print(f"Style: {style_name}, FID: {fid:.4f}")
    print("-" * 20)

