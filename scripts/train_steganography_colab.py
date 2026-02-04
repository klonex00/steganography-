import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from train_steganography import SteganographyDataset, calculate_metrics, SteganoModel
from comparison_models import UNetSteganoModel, GANSteganoModel
import os
from tqdm import tqdm
import torchvision.utils as vutils
from google.colab import drive
import shutil
import json
from datetime import datetime

# Mount Google Drive
drive.mount('/content/drive')

# Configuration
class Config:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_epochs = 50
        self.batch_size = 32
        self.image_size = (128, 128)
        self.save_dir = '/content/drive/MyDrive/steganography_results'
        self.dataset_path = '/content/drive/MyDrive/steganography_dataset/processed'
        self.checkpoint_dir = os.path.join(self.save_dir, 'checkpoints')
        
        # Create necessary directories
        os.makedirs(self.save_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # Save configuration
        self.config_path = os.path.join(self.save_dir, 'config.json')
        self.save_config()

    def save_config(self):
        config_dict = {
            'device': str(self.device),
            'num_epochs': self.num_epochs,
            'batch_size': self.batch_size,
            'image_size': self.image_size,
            'save_dir': self.save_dir,
            'dataset_path': self.dataset_path,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        with open(self.config_path, 'w') as f:
            json.dump(config_dict, f, indent=4)

def train_model(model, train_loader, val_loader, config, model_name):
    """Generic training function for all models"""
    if model_name == 'GAN':
        optimizer_G = optim.Adam(list(model.encoder.parameters()) + list(model.decoder.parameters()), 
                               lr=0.0002, betas=(0.5, 0.999))
        optimizer_D = optim.Adam(model.discriminator.parameters(), 
                               lr=0.0002, betas=(0.5, 0.999))
        adversarial_criterion = nn.BCELoss()
    else:
        optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    criterion = nn.MSELoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer if model_name != 'GAN' else optimizer_G, 
                                                   'min', patience=5)
    
    train_losses = {'total': [], 'cover': [], 'secret': []}
    val_losses = {'total': [], 'cover': [], 'secret': []}
    train_metrics = {'cover': {'psnr': [], 'ssim': []}, 'secret': {'psnr': [], 'ssim': []}}
    val_metrics = {'cover': {'psnr': [], 'ssim': []}, 'secret': {'psnr': [], 'ssim': []}}
    best_val_loss = float('inf')
    
    for epoch in range(config.num_epochs):
        # Training
        model.train()
        epoch_train_losses = {'total': 0, 'cover': 0, 'secret': 0}
        epoch_train_metrics = {'cover': {'psnr': 0, 'ssim': 0}, 'secret': {'psnr': 0, 'ssim': 0}}
        
        for cover, secret in tqdm(train_loader, desc=f'{model_name} - Epoch {epoch+1}/{config.num_epochs}'):
            cover, secret = cover.to(config.device), secret.to(config.device)
            
            if model_name == 'GAN':
                # Train Discriminator
                optimizer_D.zero_grad()
                batch_size = cover.size(0)
                
                # Real images
                label_real = torch.ones(batch_size, 1, 1, 1).to(config.device)
                output_real = model.discriminator(cover)
                d_loss_real = adversarial_criterion(output_real, label_real)
                
                # Fake images
                stego, _ = model(cover, secret)
                label_fake = torch.zeros(batch_size, 1, 1, 1).to(config.device)
                output_fake = model.discriminator(stego.detach())
                d_loss_fake = adversarial_criterion(output_fake, label_fake)
                
                d_loss = d_loss_real + d_loss_fake
                d_loss.backward()
                optimizer_D.step()
                
                # Train Generator
                optimizer_G.zero_grad()
                stego, revealed = model(cover, secret)
                
                # Adversarial loss
                output_fake = model.discriminator(stego)
                g_loss_adv = adversarial_criterion(output_fake, label_real)
                
                # Reconstruction loss
                stego_loss = criterion(stego, cover)
                secret_loss = criterion(revealed, secret)
                g_loss = g_loss_adv + stego_loss + 2.0 * secret_loss
                
                g_loss.backward()
                optimizer_G.step()
                
                loss = g_loss.item()
                epoch_train_losses['total'] += loss
                epoch_train_losses['cover'] += stego_loss.item()
                epoch_train_losses['secret'] += secret_loss.item()
            else:
                optimizer.zero_grad()
                stego, revealed = model(cover, secret)
                
                stego_loss = criterion(stego, cover)
                secret_loss = criterion(revealed, secret)
                loss = stego_loss + 2.0 * secret_loss
                
                loss.backward()
                optimizer.step()
                
                epoch_train_losses['total'] += loss.item()
                epoch_train_losses['cover'] += stego_loss.item()
                epoch_train_losses['secret'] += secret_loss.item()
            
            # Calculate metrics
            with torch.no_grad():
                metrics = calculate_metrics(cover, stego)
                for metric in ['psnr', 'ssim']:
                    epoch_train_metrics['cover'][metric] += metrics[metric]
                
                metrics = calculate_metrics(secret, revealed)
                for metric in ['psnr', 'ssim']:
                    epoch_train_metrics['secret'][metric] += metrics[metric]
        
        # Average training losses and metrics
        num_batches = len(train_loader)
        for key in epoch_train_losses:
            epoch_train_losses[key] /= num_batches
            train_losses[key].append(epoch_train_losses[key])
        
        for img_type in ['cover', 'secret']:
            for metric in ['psnr', 'ssim']:
                epoch_train_metrics[img_type][metric] /= num_batches
                train_metrics[img_type][metric].append(epoch_train_metrics[img_type][metric])
        
        # Validation
        model.eval()
        epoch_val_losses = {'total': 0, 'cover': 0, 'secret': 0}
        epoch_val_metrics = {'cover': {'psnr': 0, 'ssim': 0}, 'secret': {'psnr': 0, 'ssim': 0}}
        
        with torch.no_grad():
            for cover, secret in val_loader:
                cover, secret = cover.to(config.device), secret.to(config.device)
                stego, revealed = model(cover, secret)
                
                stego_loss = criterion(stego, cover)
                secret_loss = criterion(revealed, secret)
                loss = stego_loss + 2.0 * secret_loss
                
                epoch_val_losses['total'] += loss.item()
                epoch_val_losses['cover'] += stego_loss.item()
                epoch_val_losses['secret'] += secret_loss.item()
                
                # Calculate metrics
                metrics = calculate_metrics(cover, stego)
                for metric in ['psnr', 'ssim']:
                    epoch_val_metrics['cover'][metric] += metrics[metric]
                
                metrics = calculate_metrics(secret, revealed)
                for metric in ['psnr', 'ssim']:
                    epoch_val_metrics['secret'][metric] += metrics[metric]
        
        # Average validation losses and metrics
        num_batches = len(val_loader)
        for key in epoch_val_losses:
            epoch_val_losses[key] /= num_batches
            val_losses[key].append(epoch_val_losses[key])
        
        for img_type in ['cover', 'secret']:
            for metric in ['psnr', 'ssim']:
                epoch_val_metrics[img_type][metric] /= num_batches
                val_metrics[img_type][metric].append(epoch_val_metrics[img_type][metric])
        
        # Save best model
        if epoch_val_losses['total'] < best_val_loss:
            best_val_loss = epoch_val_losses['total']
            checkpoint_path = os.path.join(config.checkpoint_dir, f'best_{model_name.lower()}_model.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict() if model_name != 'GAN' else optimizer_G.state_dict(),
                'val_loss': best_val_loss,
            }, checkpoint_path)
            print(f'Saved best {model_name} model checkpoint to {checkpoint_path}')
        
        # Update learning rate
        scheduler.step(epoch_val_losses['total'])
        
        # Visualize results every 5 epochs
        if (epoch + 1) % 5 == 0:
            visualize_results(model, val_loader, config, epoch, model_name)
        
        # Print epoch summary
        print(f'\n{model_name} - Epoch {epoch+1}/{config.num_epochs}:')
        print(f'Train Loss: {epoch_train_losses["total"]:.4f} (Cover: {epoch_train_losses["cover"]:.4f}, Secret: {epoch_train_losses["secret"]:.4f})')
        print(f'Val Loss: {epoch_val_losses["total"]:.4f} (Cover: {epoch_val_losses["cover"]:.4f}, Secret: {epoch_val_losses["secret"]:.4f})')
        print(f'Train PSNR - Cover: {epoch_train_metrics["cover"]["psnr"]:.2f}, Secret: {epoch_train_metrics["secret"]["psnr"]:.2f}')
        print(f'Val PSNR - Cover: {epoch_val_metrics["cover"]["psnr"]:.2f}, Secret: {epoch_val_metrics["secret"]["psnr"]:.2f}')
    
    # Save training history
    history = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_metrics': train_metrics,
        'val_metrics': val_metrics
    }
    history_path = os.path.join(config.save_dir, f'{model_name.lower()}_training_history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=4)
    
    return history

def visualize_results(model, dataloader, config, epoch, model_name):
    model.eval()
    with torch.no_grad():
        cover, secret = next(iter(dataloader))
        cover, secret = cover.to(config.device), secret.to(config.device)
        
        stego, revealed = model(cover, secret)
        
        # Create visualization grid
        grid = torch.cat([cover, secret, stego, revealed], dim=0)
        grid = vutils.make_grid(grid, nrow=cover.size(0), normalize=True)
        
        plt.figure(figsize=(15, 15))
        plt.imshow(np.transpose(grid.cpu().numpy(), (1, 2, 0)))
        plt.axis('off')
        plt.title(f'{model_name} - Epoch {epoch+1} Results\nTop: Cover, Secret, Stego, Revealed')
        
        # Save to Google Drive
        save_path = os.path.join(config.save_dir, f'{model_name.lower()}_results_epoch_{epoch+1}.png')
        plt.savefig(save_path)
        plt.close()

def plot_training_curves(history, model_name, config):
    plt.figure(figsize=(15, 10))
    
    # Plot losses
    plt.subplot(2, 2, 1)
    plt.plot(history['train_losses']['total'], label='Total Train Loss')
    plt.plot(history['val_losses']['total'], label='Total Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'{model_name} - Total Loss')
    plt.legend()
    
    plt.subplot(2, 2, 2)
    plt.plot(history['train_losses']['cover'], label='Cover Train Loss')
    plt.plot(history['val_losses']['cover'], label='Cover Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'{model_name} - Cover Image Loss')
    plt.legend()
    
    plt.subplot(2, 2, 3)
    plt.plot(history['train_metrics']['cover']['psnr'], label='Cover Train PSNR')
    plt.plot(history['val_metrics']['cover']['psnr'], label='Cover Val PSNR')
    plt.xlabel('Epoch')
    plt.ylabel('PSNR (dB)')
    plt.title(f'{model_name} - Cover Image PSNR')
    plt.legend()
    
    plt.subplot(2, 2, 4)
    plt.plot(history['train_metrics']['secret']['psnr'], label='Secret Train PSNR')
    plt.plot(history['val_metrics']['secret']['psnr'], label='Secret Val PSNR')
    plt.xlabel('Epoch')
    plt.ylabel('PSNR (dB)')
    plt.title(f'{model_name} - Secret Image PSNR')
    plt.legend()
    
    plt.tight_layout()
    save_path = os.path.join(config.save_dir, f'{model_name.lower()}_training_curves.png')
    plt.savefig(save_path)
    plt.close()

def main():
    # Initialize configuration
    config = Config()
    print(f"Using device: {config.device}")
    print(f"Dataset path: {config.dataset_path}")
    
    # Verify dataset path exists
    if not os.path.exists(config.dataset_path):
        print(f"Error: Dataset path {config.dataset_path} does not exist!")
        return
    
    # Print dataset structure
    print("\nDataset structure:")
    for split in ['train', 'val', 'test']:
        split_path = os.path.join(config.dataset_path, split)
        if os.path.exists(split_path):
            n_files = len(os.listdir(split_path))
            print(f"{split}: {n_files} images")
    
    # Load dataset
    train_dataset = SteganographyDataset(config.dataset_path, split='train', target_size=config.image_size)
    val_dataset = SteganographyDataset(config.dataset_path, split='val', target_size=config.image_size)
    
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size)
    
    print(f"\nFound {len(train_dataset)} images in train set")
    print(f"Found {len(val_dataset)} images in val set")
    
    # Train all models
    models = {
        'Original': SteganoModel(),
        'UNet': UNetSteganoModel(),
        'GAN': GANSteganoModel()
    }
    
    for model_name, model in models.items():
        print(f"\nTraining {model_name} model...")
        model = model.to(config.device)
        history = train_model(model, train_loader, val_loader, config, model_name)
        plot_training_curves(history, model_name, config)
        print(f"Completed training {model_name} model")

if __name__ == '__main__':
    main() 