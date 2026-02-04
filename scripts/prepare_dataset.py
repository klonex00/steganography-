import os
import shutil
from PIL import Image
import random
from tqdm import tqdm
import glob

def process_image(img_path, target_path, image_size=(256, 256)):
    """Process a single image and save it to target path."""
    try:
        # Open and convert image
        img = Image.open(img_path)
        
        # Convert to RGB if necessary
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Resize image
        img = img.resize(image_size, Image.Resampling.LANCZOS)
        
        # Save processed image
        img.save(target_path, quality=95)
        return True
    except Exception as e:
        print(f"Error processing {img_path}: {e}")
        return False

def prepare_dataset(source_dir, target_dir, image_size=(256, 256)):
    # Create target directories
    splits = ['train', 'test', 'val']
    for split in splits:
        os.makedirs(os.path.join(target_dir, split), exist_ok=True)
    
    # Process each split
    total_processed = 0
    for split in splits:
        print(f"\nProcessing {split} split...")
        
        # Get all image files in the split directory
        split_dir = os.path.join(source_dir, split)
        if not os.path.exists(split_dir):
            print(f"Warning: {split} directory not found in {source_dir}")
            continue
            
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            image_files.extend(glob.glob(os.path.join(split_dir, '**', ext), recursive=True))
            image_files.extend(glob.glob(os.path.join(split_dir, '**', ext.upper()), recursive=True))
        
        print(f"Found {len(image_files)} images in {split} directory")
        
        # Process and save images
        processed_count = 0
        for img_path in tqdm(image_files, desc=f"Processing {split} images"):
            # Generate new filename
            filename = os.path.basename(img_path)
            new_path = os.path.join(target_dir, split, filename)
            
            # Process and save image
            if process_image(img_path, new_path, image_size):
                processed_count += 1
        
        total_processed += processed_count
        print(f"Successfully processed {processed_count} images for {split} split")
    
    print(f"\nDataset preparation complete!")
    print(f"Processed images saved to: {target_dir}")
    print(f"Total images processed: {total_processed}")

def main():
    # Configuration
    config = {
        'source_dir': './archive',  # Directory containing downloaded dataset
        'target_dir': './dataset',  # Directory for processed dataset
        'image_size': (256, 256)    # Target image size
    }
    
    # Check if source directory exists
    if not os.path.exists(config['source_dir']):
        print(f"Error: Source directory '{config['source_dir']}' not found!")
        print("Please make sure the dataset is extracted to the 'archive' directory.")
        return
    
    # Prepare dataset
    prepare_dataset(
        config['source_dir'],
        config['target_dir'],
        config['image_size']
    )

if __name__ == "__main__":
    main() 