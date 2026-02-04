import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
from tqdm import tqdm
import numpy as np
from unet_model import UNetSteganoModel
import matplotlib.pyplot as plt
import random
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import torch.nn.functional as F

class SteganographyDataset(Dataset):
    def __init__(self, data_dir, split='train', transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.split = split
        
        # Get all image files
        self.image_files = [f for f in os.listdir(os.path.join(data_dir, split)) if f.endswith('.png')]
        
        # Group images by their base name (before the suffix)
        self.image_groups = {}
        for img in self.image_files:
            base_name = '_'.join(img.split('_')[:-1])  # Remove the last part (suffix)
            if base_name not in self.image_groups:
                self.image_groups[base_name] = []
            self.image_groups[base_name].append(img)
        
        # Create pairs of cover and secret images
        self.pairs = []
        for base_name, images in self.image_groups.items():
            if len(images) >= 2:  # Need at least 2 images to form a pair
                # Randomly select two different images from the group
                img1, img2 = random.sample(images, 2)
                self.pairs.append((img1, img2))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        cover_img_name, secret_img_name = self.pairs[idx]
        
        cover_path = os.path.join(self.data_dir, self.split, cover_img_name)
        secret_path = os.path.join(self.data_dir, self.split, secret_img_name)
        
        cover_img = Image.open(cover_path).convert('RGB')
        secret_img = Image.open(secret_path).convert('RGB')
        
        if self.transform:
            cover_img = self.transform(cover_img)
            secret_img = self.transform(secret_img)
        
        return cover_img, secret_img

class CombinedLoss(nn.Module):
    def __init__(self):
        super(CombinedLoss, self).__init__()
        self.mse = nn.MSELoss()
        self.l1 = nn.L1Loss()
        self.ssim_loss = SSIMLoss()
        
    def forward(self, pred, target):
        mse_loss = self.mse(pred, target)
        l1_loss = self.l1(pred, target)
        ssim_loss = 1 - self.ssim_loss(pred, target)  # Convert SSIM to loss
        return mse_loss + 0.5 * l1_loss + 0.5 * ssim_loss

class SSIMLoss(nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 3
        self.window = self.create_window(window_size, self.channel)

    def gaussian(self, window_size, sigma):
        gauss = torch.Tensor([np.exp(-(x - window_size//2)**2/float(sigma**2)) for x in range(window_size)])
        return gauss/gauss.sum()

    def create_window(self, window_size, channel):
        _1D_window = self.gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()
        window = self.window.to(img1.device)

        mu1 = F.conv2d(img1, window, padding=self.window_size//2, groups=channel)
        mu2 = F.conv2d(img2, window, padding=self.window_size//2, groups=channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(img1*img1, window, padding=self.window_size//2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2*img2, window, padding=self.window_size//2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1*img2, window, padding=self.window_size//2, groups=channel) - mu1_mu2

        C1 = 0.01**2
        C2 = 0.03**2

        ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

        if self.size_average:
            return ssim_map.mean()
        else:
            return ssim_map.mean(1).mean(1).mean(1)

def calculate_metrics(img1, img2):
    """Calculate PSNR and SSIM between two images."""
    img1_np = img1.cpu().numpy()
    img2_np = img2.cpu().numpy()
    
    # Calculate PSNR
    mse = np.mean((img1_np - img2_np) ** 2)
    if mse == 0:
        return float('inf'), 1.0
    psnr = 20 * np.log10(1.0 / np.sqrt(mse))
    
    # Calculate SSIM with adjusted window size
    # Get the minimum dimension of the image
    min_dim = min(img1_np.shape[1], img1_np.shape[2])
    # Use a smaller window size if image is small
    win_size = min(7, min_dim if min_dim % 2 == 1 else min_dim - 1)
    if win_size < 3:
        win_size = 3  # Minimum window size
    
    ssim_val = ssim(img1_np, img2_np, 
                    channel_axis=0, 
                    data_range=1.0,
                    win_size=win_size)
    
    return psnr, ssim_val

def train_model(model, train_loader, val_loader, device, num_epochs=100):  # Increased epochs
    criterion = CombinedLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=1e-4)  # Changed to AdamW with lower lr
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, 
        T_0=10,  # Restart every 10 epochs
        T_mult=2,  # Double the restart interval after each restart
        eta_min=1e-6  # Minimum learning rate
    )
    
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    best_metrics = {'psnr': 0, 'ssim': 0}
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0
        train_steps = 0
        
        for cover, secret in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]'):
            cover, secret = cover.to(device), secret.to(device)
            
            optimizer.zero_grad()
            stego, revealed = model(cover, secret)
            
            # Calculate losses with adjusted weights
            stego_loss = criterion(stego, cover)
            secret_loss = criterion(revealed, secret)
            loss = stego_loss + 3.0 * secret_loss  # Increased weight for secret image reconstruction
            
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            train_loss += loss.item()
            train_steps += 1
        
        avg_train_loss = train_loss / train_steps
        train_losses.append(avg_train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_steps = 0
        val_psnr = 0
        val_ssim = 0
        
        with torch.no_grad():
            for cover, secret in tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Val]'):
                cover, secret = cover.to(device), secret.to(device)
                stego, revealed = model(cover, secret)
                
                stego_loss = criterion(stego, cover)
                secret_loss = criterion(revealed, secret)
                loss = stego_loss + 3.0 * secret_loss
                
                val_loss += loss.item()
                val_steps += 1
                
                # Calculate metrics
                psnr_val, ssim_val = calculate_metrics(revealed, secret)
                val_psnr += psnr_val
                val_ssim += ssim_val
        
        avg_val_loss = val_loss / val_steps
        avg_val_psnr = val_psnr / val_steps
        avg_val_ssim = val_ssim / val_steps
        val_losses.append(avg_val_loss)
        
        # Learning rate scheduling
        scheduler.step()
        
        # Print current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        
        # Save best model based on validation loss and metrics
        if avg_val_loss < best_val_loss and avg_val_psnr > best_metrics['psnr']:
            best_val_loss = avg_val_loss
            best_metrics['psnr'] = avg_val_psnr
            best_metrics['ssim'] = avg_val_ssim
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_val_loss,
                'psnr': avg_val_psnr,
                'ssim': avg_val_ssim
            }, 'checkpoints/best_unet_model.pth')
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_val_loss,
                'psnr': avg_val_psnr,
                'ssim': avg_val_ssim
            }, f'checkpoints/unet_model_epoch_{epoch+1}.pth')
        
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'Training Loss: {avg_train_loss:.6f}')
        print(f'Validation Loss: {avg_val_loss:.6f}')
        print(f'Validation PSNR: {avg_val_psnr:.2f} dB')
        print(f'Validation SSIM: {avg_val_ssim:.4f}')
        print(f'Best Validation Loss: {best_val_loss:.6f}')
        print(f'Best PSNR: {best_metrics["psnr"]:.2f} dB')
        print(f'Best SSIM: {best_metrics["ssim"]:.4f}')
        print(f'Current Learning Rate: {current_lr:.6f}')
        print('-' * 50)
    
    # Plot training history
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training History')
    plt.legend()
    plt.savefig('training_history.png')
    plt.close()

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Create checkpoints directory if it doesn't exist
    os.makedirs('checkpoints', exist_ok=True)
    
    # Data transforms with enhanced augmentation
    transform = transforms.Compose([
        transforms.Resize((256, 256)),  # Increased image size
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor()
    ])
    
    # Create datasets
    train_dataset = SteganographyDataset(
        data_dir='dataset',
        split='train',
        transform=transform
    )
    
    val_dataset = SteganographyDataset(
        data_dir='dataset',
        split='val',
        transform=transforms.Compose([
            transforms.Resize((256, 256)),  # Increased image size
            transforms.ToTensor()
        ])
    )
    
    # Create data loaders with increased batch size
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
    
    # Initialize model
    model = UNetSteganoModel().to(device)
    
    # Train model
    train_model(model, train_loader, val_loader, device)

if __name__ == '__main__':
    main() 