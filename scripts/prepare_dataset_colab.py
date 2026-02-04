import os
import shutil
from PIL import Image
import torchvision.transforms as transforms
from tqdm import tqdm
import numpy as np
from google.colab import drive
import random

def create_directory_structure(base_path):
    """Create necessary directories for the dataset."""
    directories = [
        os.path.join(base_path, 'train'),
        os.path.join(base_path, 'val'),
        os.path.join(base_path, 'test')
    ]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    return directories

def preprocess_image(image_path, target_size=(128, 128), quality=95):
    """
    Preprocess a single image with high-quality settings.
    
    Args:
        image_path: Path to the input image
        target_size: Target size for the image (width, height)
        quality: JPEG quality (1-100)
    
    Returns:
        Preprocessed image as PIL Image
    """
    try:
        # Open image
        img = Image.open(image_path).convert('RGB')
        
        # Calculate aspect ratio
        width, height = img.size
        aspect_ratio = width / height
        
        # Calculate new dimensions while preserving aspect ratio
        if aspect_ratio > 1:
            new_width = target_size[0]
            new_height = int(target_size[0] / aspect_ratio)
        else:
            new_height = target_size[1]
            new_width = int(target_size[1] * aspect_ratio)
        
        # Resize image with high-quality settings
        img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # Create a new image with target size and paste the resized image
        new_img = Image.new('RGB', target_size, (255, 255, 255))
        paste_x = (target_size[0] - new_width) // 2
        paste_y = (target_size[1] - new_height) // 2
        new_img.paste(img, (paste_x, paste_y))
        
        return new_img
    except Exception as e:
        print(f"Error processing {image_path}: {str(e)}")
        return None

def get_image_files(directory):
    """Get all image files from a directory and its subdirectories."""
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    image_files = []
    
    # Check if directory exists
    if not os.path.exists(directory):
        print(f"Warning: Directory {directory} does not exist")
        return image_files
    
    # Walk through directory
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(image_extensions):
                image_files.append(os.path.join(root, file))
    
    return image_files

def prepare_dataset(source_path, target_path):
    """
    Prepare the dataset by preprocessing images and organizing them into train/val/test splits.
    
    Args:
        source_path: Path to the source dataset
        target_path: Path to save the processed dataset
    """
    # Create directory structure
    train_dir, val_dir, test_dir = create_directory_structure(target_path)
    
    # Get image files from each split
    train_files = get_image_files(os.path.join(source_path, 'train'))
    val_files = get_image_files(os.path.join(source_path, 'val'))
    test_files = get_image_files(os.path.join(source_path, 'test'))
    
    # Process and save images
    print("Processing training images...")
    for img_path in tqdm(train_files):
        processed_img = preprocess_image(img_path)
        if processed_img:
            save_path = os.path.join(train_dir, os.path.basename(img_path))
            processed_img.save(save_path, quality=95, optimize=True)
    
    print("Processing validation images...")
    for img_path in tqdm(val_files):
        processed_img = preprocess_image(img_path)
        if processed_img:
            save_path = os.path.join(val_dir, os.path.basename(img_path))
            processed_img.save(save_path, quality=95, optimize=True)
    
    print("Processing test images...")
    for img_path in tqdm(test_files):
        processed_img = preprocess_image(img_path)
        if processed_img:
            save_path = os.path.join(test_dir, os.path.basename(img_path))
            processed_img.save(save_path, quality=95, optimize=True)
    
    # Print dataset statistics
    print("\nDataset Statistics:")
    print(f"Training images: {len(train_files)}")
    print(f"Validation images: {len(val_files)}")
    print(f"Test images: {len(test_files)}")
    print(f"Total images: {len(train_files) + len(val_files) + len(test_files)}")

def main():
    # Mount Google Drive
    drive.mount('/content/drive')
    
    # Define paths
    source_path = '/content/drive/MyDrive/cnsdata'  # Updated path to cnsdata folder
    target_path = '/content/drive/MyDrive/steganography_dataset/processed'
    
    # Print paths for debugging
    print(f"Source path: {source_path}")
    print(f"Target path: {target_path}")
    
    # Check if source path exists
    if not os.path.exists(source_path):
        print(f"Error: Source path {source_path} does not exist!")
        return
    
    # List contents of source directory
    print("\nContents of source directory:")
    print(os.listdir(source_path))
    
    # Create target directory if it doesn't exist
    os.makedirs(target_path, exist_ok=True)
    
    # Prepare dataset
    print("\nStarting dataset preparation...")
    prepare_dataset(source_path, target_path)
    print("Dataset preparation completed!")

if __name__ == '__main__':
    main() 