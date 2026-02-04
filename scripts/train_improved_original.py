import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from PIL import Image
import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import argparse
from torch.optim.lr_scheduler import CosineAnnealingLR

class ImprovedSteganoModel(nn.Module):
    def __init__(self):
        super(ImprovedSteganoModel, self).__init__()
        
        # Encoder with residual connections
        self.encoder_conv1 = nn.Sequential(
            nn.Conv2d(6, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        self.encoder_conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        self.encoder_conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        self.encoder_conv4 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        self.encoder_conv5 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        self.encoder_final = nn.Sequential(
            nn.Conv2d(64, 3, kernel_size=3, padding=1),
            nn.Tanh()
        )
        
        # Decoder with skip connections
        self.decoder_conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        self.decoder_conv2 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),  # 128 = 64 + 64 (skip connection)
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        self.decoder_conv3 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),  # 256 = 128 + 128 (skip connection)
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        self.decoder_conv4 = nn.Sequential(
            nn.Conv2d(384, 128, kernel_size=3, padding=1),  # 384 = 256 + 128 (skip connection)
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        self.decoder_conv5 = nn.Sequential(
            nn.Conv2d(192, 64, kernel_size=3, padding=1),  # 192 = 128 + 64 (skip connection)
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        self.decoder_final = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 3, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Conv2d(6, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1),
            nn.Sigmoid()
        )
    
    def forward(self, cover, secret):
        # Concatenate inputs
        x = torch.cat((cover, secret), dim=1)
        
        # Apply attention
        att = self.attention(x)
        x = x * att
        
        # Encoder path with skip connections
        e1 = self.encoder_conv1(x)
        e2 = self.encoder_conv2(e1)
        e3 = self.encoder_conv3(e2)
        e4 = self.encoder_conv4(e3)
        e5 = self.encoder_conv5(e4)
        encoded = self.encoder_final(e5)
        
        # Create stego image (normalize encoded output to [0, 1] range)
        encoded_normalized = (encoded + 1) / 2  # Convert from [-1, 1] to [0, 1]
        stego = cover + encoded_normalized
        stego = torch.clamp(stego, 0, 1)
        
        # Decoder path with skip connections
        d1 = self.decoder_conv1(stego)
        d2 = self.decoder_conv2(torch.cat([d1, e5], dim=1))
        d3 = self.decoder_conv3(torch.cat([d2, e4], dim=1))
        d4 = self.decoder_conv4(torch.cat([d3, e3], dim=1))
        d5 = self.decoder_conv5(torch.cat([d4, e2], dim=1))
        revealed = self.decoder_final(d5)
        
        return stego, revealed

def prepare_dataset(data_dir, batch_size=32, image_size=128):
    """Prepare dataset for training."""
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor()
    ])
    
    class SteganographyDataset(torch.utils.data.Dataset):
        def __init__(self, data_dir, transform=None):
            self.data_dir = data_dir
            self.transform = transform
            self.image_files = [f for f in os.listdir(data_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
        
        def __len__(self):
            return len(self.image_files)
        
        def __getitem__(self, idx):
            img_path = os.path.join(self.data_dir, self.image_files[idx])
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image
    
    dataset = SteganographyDataset(data_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    return dataloader

def calculate_metrics(original, processed):
    """Calculate PSNR and SSIM metrics."""
    original_np = original.squeeze(0).cpu().numpy()
    processed_np = processed.squeeze(0).cpu().numpy()
    
    # Convert to channel-last format for SSIM
    original_np = np.transpose(original_np, (1, 2, 0))
    processed_np = np.transpose(processed_np, (1, 2, 0))
    
    psnr_value = float(psnr(original_np, processed_np, data_range=1.0))
    ssim_value = float(ssim(original_np, processed_np, channel_axis=2, data_range=1.0))
    
    return psnr_value, ssim_value

def train_model(model, train_loader, val_loader, device, num_epochs=100, learning_rate=0.0001):
    """Train the improved steganography model."""
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999))
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=learning_rate/10)
    
    # Loss functions
    mse_loss = nn.MSELoss()
    l1_loss = nn.L1Loss()
    
    # Training history
    history = {
        'train_loss': [], 'val_loss': [],
        'train_stego_psnr': [], 'val_stego_psnr': [],
        'train_stego_ssim': [], 'val_stego_ssim': [],
        'train_secret_psnr': [], 'val_secret_psnr': [],
        'train_secret_ssim': [], 'val_secret_ssim': []
    }
    
    best_val_psnr = 0.0
    os.makedirs('checkpoints', exist_ok=True)
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_metrics = {'stego': {'psnr': 0.0, 'ssim': 0.0}, 'secret': {'psnr': 0.0, 'ssim': 0.0}}
        
        train_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
        for batch_idx, cover_images in enumerate(train_bar):
            # Use random images from the same batch as secret images
            secret_images = cover_images[torch.randperm(cover_images.size(0))]
            
            cover_images = cover_images.to(device)
            secret_images = secret_images.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            stego_images, revealed_secrets = model(cover_images, secret_images)
            
            # Calculate losses with adjusted weights
            stego_loss = mse_loss(stego_images, cover_images) + 0.1 * l1_loss(stego_images, cover_images)
            secret_loss = mse_loss(revealed_secrets, secret_images) + 0.1 * l1_loss(revealed_secrets, secret_images)
            total_loss = stego_loss + 2.0 * secret_loss  # Emphasize secret reconstruction
            
            # Backward pass
            total_loss.backward()
            optimizer.step()
            
            # Calculate metrics
            with torch.no_grad():
                stego_psnr, stego_ssim = calculate_metrics(cover_images[0], stego_images[0])
                secret_psnr, secret_ssim = calculate_metrics(secret_images[0], revealed_secrets[0])
                
                train_metrics['stego']['psnr'] += stego_psnr
                train_metrics['stego']['ssim'] += stego_ssim
                train_metrics['secret']['psnr'] += secret_psnr
                train_metrics['secret']['ssim'] += secret_ssim
            
            train_loss += total_loss.item()
            
            train_bar.set_postfix({
                'loss': f'{total_loss.item():.4f}',
                'stego_psnr': f'{stego_psnr:.2f}',
                'secret_psnr': f'{secret_psnr:.2f}'
            })
        
        # Calculate average training metrics
        train_loss /= len(train_loader)
        for key in train_metrics:
            for metric in train_metrics[key]:
                train_metrics[key][metric] /= len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_metrics = {'stego': {'psnr': 0.0, 'ssim': 0.0}, 'secret': {'psnr': 0.0, 'ssim': 0.0}}
        
        with torch.no_grad():
            val_bar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Val]')
            for batch_idx, cover_images in enumerate(val_bar):
                secret_images = cover_images[torch.randperm(cover_images.size(0))]
                
                cover_images = cover_images.to(device)
                secret_images = secret_images.to(device)
                
                # Forward pass
                stego_images, revealed_secrets = model(cover_images, secret_images)
                
                # Calculate losses
                stego_loss = mse_loss(stego_images, cover_images) + 0.1 * l1_loss(stego_images, cover_images)
                secret_loss = mse_loss(revealed_secrets, secret_images) + 0.1 * l1_loss(revealed_secrets, secret_images)
                total_loss = stego_loss + 2.0 * secret_loss
                
                # Calculate metrics
                stego_psnr, stego_ssim = calculate_metrics(cover_images[0], stego_images[0])
                secret_psnr, secret_ssim = calculate_metrics(secret_images[0], revealed_secrets[0])
                
                val_metrics['stego']['psnr'] += stego_psnr
                val_metrics['stego']['ssim'] += stego_ssim
                val_metrics['secret']['psnr'] += secret_psnr
                val_metrics['secret']['ssim'] += secret_ssim
                
                val_loss += total_loss.item()
                
                val_bar.set_postfix({
                    'loss': f'{total_loss.item():.4f}',
                    'stego_psnr': f'{stego_psnr:.2f}',
                    'secret_psnr': f'{secret_psnr:.2f}'
                })
        
        # Calculate average validation metrics
        val_loss /= len(val_loader)
        for key in val_metrics:
            for metric in val_metrics[key]:
                val_metrics[key][metric] /= len(val_loader)
        
        # Update learning rate
        scheduler.step()
        
        # Save history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_stego_psnr'].append(train_metrics['stego']['psnr'])
        history['val_stego_psnr'].append(val_metrics['stego']['psnr'])
        history['train_stego_ssim'].append(train_metrics['stego']['ssim'])
        history['val_stego_ssim'].append(val_metrics['stego']['ssim'])
        history['train_secret_psnr'].append(train_metrics['secret']['psnr'])
        history['val_secret_psnr'].append(val_metrics['secret']['psnr'])
        history['train_secret_ssim'].append(train_metrics['secret']['ssim'])
        history['val_secret_ssim'].append(val_metrics['secret']['ssim'])
        
        # Save best model
        avg_val_psnr = (val_metrics['stego']['psnr'] + val_metrics['secret']['psnr']) / 2
        if avg_val_psnr > best_val_psnr:
            best_val_psnr = avg_val_psnr
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_psnr': best_val_psnr,
                'history': history
            }, 'checkpoints/best_improved_original_model.pth')
            print(f"\nSaved new best model with validation PSNR: {best_val_psnr:.2f}")
        
        # Print epoch summary
        print(f"\nEpoch {epoch+1}/{num_epochs} Summary:")
        print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        print(f"Train - Stego PSNR: {train_metrics['stego']['psnr']:.2f}, SSIM: {train_metrics['stego']['ssim']:.4f}")
        print(f"Train - Secret PSNR: {train_metrics['secret']['psnr']:.2f}, SSIM: {train_metrics['secret']['ssim']:.4f}")
        print(f"Val - Stego PSNR: {val_metrics['stego']['psnr']:.2f}, SSIM: {val_metrics['stego']['ssim']:.4f}")
        print(f"Val - Secret PSNR: {val_metrics['secret']['psnr']:.2f}, SSIM: {val_metrics['secret']['ssim']:.4f}")
    
    return history

def plot_training_history(history):
    """Plot training history."""
    plt.figure(figsize=(15, 10))
    
    # Plot losses
    plt.subplot(2, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot PSNR
    plt.subplot(2, 2, 2)
    plt.plot(history['train_stego_psnr'], label='Train Stego PSNR')
    plt.plot(history['val_stego_psnr'], label='Val Stego PSNR')
    plt.plot(history['train_secret_psnr'], label='Train Secret PSNR')
    plt.plot(history['val_secret_psnr'], label='Val Secret PSNR')
    plt.title('PSNR Metrics')
    plt.xlabel('Epoch')
    plt.ylabel('PSNR (dB)')
    plt.legend()
    
    # Plot SSIM
    plt.subplot(2, 2, 3)
    plt.plot(history['train_stego_ssim'], label='Train Stego SSIM')
    plt.plot(history['val_stego_ssim'], label='Val Stego SSIM')
    plt.plot(history['train_secret_ssim'], label='Train Secret SSIM')
    plt.plot(history['val_secret_ssim'], label='Val Secret SSIM')
    plt.title('SSIM Metrics')
    plt.xlabel('Epoch')
    plt.ylabel('SSIM')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_history_improved.png')
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Train Improved Steganography Model')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to dataset directory')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of epochs to train')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--image_size', type=int, default=128, help='Size of input images')
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Prepare dataset
    print("Preparing dataset...")
    train_loader = prepare_dataset(args.data_dir, args.batch_size, args.image_size)
    
    # Create validation loader (using same data for simplicity)
    val_loader = prepare_dataset(args.data_dir, args.batch_size, args.image_size)
    
    # Create and train model
    print("Creating model...")
    model = ImprovedSteganoModel()
    
    print("Starting training...")
    history = train_model(
        model, train_loader, val_loader, device,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate
    )
    
    # Plot training history
    print("Plotting training history...")
    plot_training_history(history)
    
    print("Training completed!")

if __name__ == "__main__":
    main() 