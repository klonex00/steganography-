import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, datasets
import os
import glob
import random
import matplotlib.pyplot as plt
from PIL import Image
import time
import numpy as np
from skimage.metrics import structural_similarity as ssim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import json
from tqdm import tqdm
from datetime import datetime, timedelta
import torchvision
import torch.nn.functional as F

# --- Model Architecture ---

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(6, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 3, kernel_size=3, padding=1),
            nn.Tanh()  # Constrain output to [-1, 1]
        )
    
    def forward(self, x):
        return self.encoder_conv(x)

class Decoder(nn.Module):
    """Neural network for decoding a secret image from a stego image."""
    def __init__(self):
        super(Decoder, self).__init__()
        self.decoder_conv = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 3, kernel_size=3, padding=1),
            nn.Sigmoid()  # Output in [0, 1]
        )
    
    def forward(self, x):
        return self.decoder_conv(x)

class SteganoModel(nn.Module):
    """Combined model for both hiding and revealing images."""
    def __init__(self):
        super(SteganoModel, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        
    def forward(self, cover, secret):
        # Hiding process
        x = torch.cat((cover, secret), dim=1)  # Concatenate along channel dimension
        encoded_image = self.encoder(x)         # Generate encoding
        stego_image = cover + encoded_image     # Add encoding to cover image
        stego_image = torch.clamp(stego_image, 0, 1)  # Ensure valid image range
        
        # Revealing process
        revealed_secret = self.decoder(stego_image)
        revealed_secret = torch.clamp(revealed_secret, 0, 1)  # Ensure valid image range
        
        return stego_image, revealed_secret

# --- Enhanced Dataset Class ---

class SteganographyDataset(Dataset):
    """Enhanced dataset for training steganography models with data augmentation."""
    def __init__(self, image_dir, split='train', transform=None, target_size=(128, 128), max_samples=None):
        """
        Args:
            image_dir (str): Directory containing the dataset
            split (str): One of 'train', 'val', or 'test'
            transform (callable, optional): Transform to apply to images
            target_size (tuple): Size to resize images to
        """
        self.image_dir = os.path.join(image_dir, split)
        self.image_paths = []
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            self.image_paths.extend(glob.glob(os.path.join(self.image_dir, ext)))
            self.image_paths.extend(glob.glob(os.path.join(self.image_dir, ext.upper())))
        
        if len(self.image_paths) == 0:
            raise ValueError(f"No images found in {self.image_dir}")
            
        # Limit number of samples if specified
        if max_samples is not None:
            self.image_paths = self.image_paths[:max_samples]
            
        print(f"Found {len(self.image_paths)} images in {split} set")
        
        if transform is None:
            if split == 'train':
                # Simplified training transforms
                self.transform = transforms.Compose([
                    transforms.Resize(target_size),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                ])
            else:
                # Simplified validation transforms
                self.transform = transforms.Compose([
                    transforms.Resize(target_size),
                    transforms.ToTensor(),
                ])
        else:
            self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Get cover image
        cover_path = self.image_paths[idx]
        cover_img = Image.open(cover_path).convert('RGB')
        
        # Get a random different image for secret
        secret_idx = random.randint(0, len(self.image_paths) - 1)
        while secret_idx == idx:  # Ensure different images
            secret_idx = random.randint(0, len(self.image_paths) - 1)
        
        secret_path = self.image_paths[secret_idx]
        secret_img = Image.open(secret_path).convert('RGB')
        
        # Apply transformations
        if self.transform:
            cover_tensor = self.transform(cover_img)
            secret_tensor = self.transform(secret_img)
        
        return cover_tensor, secret_tensor

# --- Enhanced Training Functions ---

def calculate_metrics(original, generated):
    """Calculate PSNR, SSIM, and MSE metrics."""
    # Convert to numpy arrays with detach
    original_np = original.detach().cpu().numpy()
    generated_np = generated.detach().cpu().numpy()
    
    # Calculate MSE
    mse = np.mean((original_np - generated_np) ** 2)
    
    # Calculate PSNR
    psnr = 20 * np.log10(1.0 / np.sqrt(mse))
    
    # Calculate SSIM for each image in the batch
    ssim_values = []
    for i in range(original_np.shape[0]):
        # Transpose to (H, W, C) format
        orig_img = original_np[i].transpose(1, 2, 0)
        gen_img = generated_np[i].transpose(1, 2, 0)
        
        # Calculate SSIM with appropriate window size
        ssim_value = ssim(orig_img, gen_img, 
                         channel_axis=2,  # Specify channel axis
                         data_range=1.0,
                         win_size=3)  # Use smaller window size
        ssim_values.append(ssim_value)
    
    # Average SSIM across batch
    ssim_value = np.mean(ssim_values)
    
    return {
        'mse': float(mse),
        'psnr': float(psnr),
        'ssim': float(ssim_value)
    }

def perceptual_loss(x, y):
    """Calculate perceptual loss using VGG features."""
    vgg = torchvision.models.vgg16(pretrained=True).features[:16].to(x.device)
    vgg.eval()
    
    # Extract features
    x_features = vgg(x)
    y_features = vgg(y)
    
    # Calculate MSE between features
    return F.mse_loss(x_features, y_features)

def train_one_epoch(model, dataloader, optimizer, criterion, device, epoch, pbar=None):
    """Enhanced training for one epoch with metrics and progress bar."""
    model.train()
    running_loss = 0.0
    running_cover_loss = 0.0
    running_secret_loss = 0.0
    running_metrics = {'cover': {'mse': 0.0, 'psnr': 0.0, 'ssim': 0.0},
                      'secret': {'mse': 0.0, 'psnr': 0.0, 'ssim': 0.0}}
    
    start_time = time.time()
    
    for i, (cover_images, secret_images) in enumerate(dataloader):
        cover_images = cover_images.to(device)
        secret_images = secret_images.to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        stego_images, revealed_secrets = model(cover_images, secret_images)
        
        # Calculate losses with adjusted weights
        stego_loss = criterion(stego_images, cover_images)
        secret_loss = criterion(revealed_secrets, secret_images)
        
        # Combined loss with emphasis on secret reconstruction
        loss = stego_loss + 2.0 * secret_loss
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        # Update statistics
        running_loss += loss.item()
        running_cover_loss += stego_loss.item()
        running_secret_loss += secret_loss.item()
        
        # Calculate metrics
        cover_metrics = calculate_metrics(cover_images, stego_images)
        secret_metrics = calculate_metrics(secret_images, revealed_secrets)
        
        for metric in ['mse', 'psnr', 'ssim']:
            running_metrics['cover'][metric] += cover_metrics[metric]
            running_metrics['secret'][metric] += secret_metrics[metric]
        
        # Update progress bar
        if pbar is not None:
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'cover_psnr': f"{cover_metrics['psnr']:.2f}",
                'secret_psnr': f"{secret_metrics['psnr']:.2f}"
            })
            pbar.update(1)
    
    # Calculate averages
    num_batches = len(dataloader)
    avg_loss = running_loss / num_batches
    avg_cover_loss = running_cover_loss / num_batches
    avg_secret_loss = running_secret_loss / num_batches
    
    for metric in ['mse', 'psnr', 'ssim']:
        running_metrics['cover'][metric] /= num_batches
        running_metrics['secret'][metric] /= num_batches
    
    epoch_time = time.time() - start_time
    
    return {
        'loss': avg_loss,
        'cover_loss': avg_cover_loss,
        'secret_loss': avg_secret_loss,
        'metrics': running_metrics,
        'time': epoch_time
    }

def validate(model, dataloader, criterion, device, pbar=None):
    """Enhanced validation with metrics and progress bar."""
    model.eval()
    running_loss = 0.0
    running_cover_loss = 0.0
    running_secret_loss = 0.0
    running_metrics = {'cover': {'mse': 0.0, 'psnr': 0.0, 'ssim': 0.0},
                      'secret': {'mse': 0.0, 'psnr': 0.0, 'ssim': 0.0}}
    
    with torch.no_grad():
        for cover_images, secret_images in dataloader:
            cover_images = cover_images.to(device)
            secret_images = secret_images.to(device)
            
            # Forward pass
            stego_images, revealed_secrets = model(cover_images, secret_images)
            
            # Calculate losses
            stego_loss = criterion(stego_images, cover_images)
            secret_loss = criterion(revealed_secrets, secret_images)
            loss = stego_loss + 1.5 * secret_loss
            
            # Update statistics
            running_loss += loss.item()
            running_cover_loss += stego_loss.item()
            running_secret_loss += secret_loss.item()
            
            # Calculate metrics
            cover_metrics = calculate_metrics(cover_images, stego_images)
            secret_metrics = calculate_metrics(secret_images, revealed_secrets)
            
            for metric in ['mse', 'psnr', 'ssim']:
                running_metrics['cover'][metric] += cover_metrics[metric]
                running_metrics['secret'][metric] += secret_metrics[metric]
            
            # Update progress bar
            if pbar is not None:
                pbar.update(1)
    
    # Calculate averages
    num_batches = len(dataloader)
    avg_loss = running_loss / num_batches
    avg_cover_loss = running_cover_loss / num_batches
    avg_secret_loss = running_secret_loss / num_batches
    
    for metric in ['mse', 'psnr', 'ssim']:
        running_metrics['cover'][metric] /= num_batches
        running_metrics['secret'][metric] /= num_batches
    
    return {
        'loss': avg_loss,
        'cover_loss': avg_cover_loss,
        'secret_loss': avg_secret_loss,
        'metrics': running_metrics
    }

def save_checkpoint(model, optimizer, epoch, metrics, is_best, save_dir, filename='checkpoint.pth'):
    """Save model checkpoint with training state."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics,
        'is_best': is_best
    }
    
    # Save regular checkpoint
    checkpoint_path = os.path.join(save_dir, filename)
    torch.save(checkpoint, checkpoint_path)
    
    # If this is the best model, save a copy
    if is_best:
        best_path = os.path.join(save_dir, 'best_model.pth')
        torch.save(checkpoint, best_path)
    
    # Save epoch-specific checkpoint
    epoch_path = os.path.join(save_dir, f'checkpoint_epoch_{epoch:03d}.pth')
    torch.save(checkpoint, epoch_path)
    
    # Keep only the last 5 checkpoints to save disk space
    checkpoints = sorted(glob.glob(os.path.join(save_dir, 'checkpoint_epoch_*.pth')))
    if len(checkpoints) > 5:
        for old_checkpoint in checkpoints[:-5]:
            try:
                os.remove(old_checkpoint)
            except Exception as e:
                print(f"Warning: Could not remove old checkpoint {old_checkpoint}: {e}")

def load_checkpoint(model, optimizer, checkpoint_path, device):
    """Load model checkpoint and return training state."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['epoch'], checkpoint['metrics'], checkpoint['is_best']

def main():
    # Configuration
    config = {
        'image_dir': './dataset',
        'batch_size': 32,          # Increased batch size
        'num_epochs': 50,          # Reduced epochs
        'learning_rate': 0.001,    # Increased learning rate
        'target_size': (128, 128),
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'save_dir': './checkpoints',
        'early_stopping_patience': 5,
        'min_delta': 0.001,
        'num_workers': 4,
        'checkpoint_interval': 5,
        'resume_training': False,
        'checkpoint_path': None,
        'max_samples': 1000
    }
    
    # Create save directory
    os.makedirs(config['save_dir'], exist_ok=True)
    
    # Save configuration
    with open(os.path.join(config['save_dir'], 'config.json'), 'w') as f:
        json.dump(config, f, indent=4)
    
    # Convert device string to torch.device
    device = torch.device(config['device'])
    
    # Create datasets with limited samples
    train_dataset = SteganographyDataset(
        config['image_dir'],
        split='train',
        target_size=config['target_size'],
        max_samples=config['max_samples']
    )
    
    val_dataset = SteganographyDataset(
        config['image_dir'],
        split='val',
        target_size=config['target_size'],
        max_samples=config['max_samples'] // 5  # 20% of training samples for validation
    )
    
    # Create data loaders with optimized settings for CPU
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=True
    )
    
    # Initialize model
    model = SteganoModel().to(device)
    
    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    
    # Learning rate scheduler
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=3
    )
    
    # Training loop variables
    start_epoch = 0
    best_val_loss = float('inf')
    patience_counter = 0
    history = {'train': [], 'val': []}
    
    # Resume training if specified
    if config['resume_training']:
        if config['checkpoint_path'] is None:
            # Find the latest checkpoint
            checkpoints = sorted(glob.glob(os.path.join(config['save_dir'], 'checkpoint_epoch_*.pth')))
            if checkpoints:
                config['checkpoint_path'] = checkpoints[-1]
        
        if config['checkpoint_path'] and os.path.exists(config['checkpoint_path']):
            print(f"\nResuming training from checkpoint: {config['checkpoint_path']}")
            start_epoch, checkpoint_metrics, is_best = load_checkpoint(
                model, optimizer, config['checkpoint_path'], device
            )
            best_val_loss = checkpoint_metrics['val_loss'] if is_best else float('inf')
            history = checkpoint_metrics.get('history', {'train': [], 'val': []})
            print(f"Resumed from epoch {start_epoch}")
    
    print("\nStarting training...")
    print(f"Device: {device}")
    print(f"Batch size: {config['batch_size']}")
    print(f"Number of epochs: {config['num_epochs']}")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Checkpoint interval: Every {config['checkpoint_interval']} epochs")
    
    # Create overall progress bar
    overall_pbar = tqdm(total=config['num_epochs'] - start_epoch, 
                       initial=start_epoch,
                       desc="Overall Progress", 
                       position=0)
    
    for epoch in range(start_epoch, config['num_epochs']):
        # Create epoch progress bar
        epoch_pbar = tqdm(total=len(train_loader), 
                         desc=f"Epoch {epoch+1}/{config['num_epochs']}", 
                         position=1)
        
        # Train
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, criterion, device, epoch, epoch_pbar
        )
        history['train'].append(train_metrics)
        
        # Close epoch progress bar
        epoch_pbar.close()
        
        # Create validation progress bar
        val_pbar = tqdm(total=len(val_loader), desc="Validation", position=1)
        
        # Validate
        val_metrics = validate(
            model, val_loader, criterion, device, val_pbar
        )
        history['val'].append(val_metrics)
        
        # Close validation progress bar
        val_pbar.close()
        
        # Learning rate scheduling
        scheduler.step(val_metrics['loss'])
        
        # Check if this is the best model
        is_best = val_metrics['loss'] < best_val_loss - config['min_delta']
        if is_best:
            best_val_loss = val_metrics['loss']
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Save checkpoint
        if (epoch + 1) % config['checkpoint_interval'] == 0 or is_best:
            metrics = {
                'train_loss': train_metrics['loss'],
                'val_loss': val_metrics['loss'],
                'best_val_loss': best_val_loss,
                'history': history
            }
            save_checkpoint(
                model, optimizer, epoch + 1, metrics, is_best,
                config['save_dir'], 'checkpoint.pth'
            )
            if is_best:
                print(f"\nSaved best model with validation loss: {best_val_loss:.4f}")
        
        # Update overall progress
        overall_pbar.update(1)
        overall_pbar.set_postfix({
            'train_loss': f"{train_metrics['loss']:.4f}",
            'val_loss': f"{val_metrics['loss']:.4f}",
            'best_val_loss': f"{best_val_loss:.4f}"
        })
        
        # Early stopping
        if patience_counter >= config['early_stopping_patience']:
            print(f"\nEarly stopping triggered after {epoch+1} epochs")
            break
    
    # Close overall progress bar
    overall_pbar.close()
    
    # Save final training history
    with open(os.path.join(config['save_dir'], 'history.json'), 'w') as f:
        json.dump(history, f, indent=4)
    
    print("\nTraining completed!")

if __name__ == "__main__":
    main() 