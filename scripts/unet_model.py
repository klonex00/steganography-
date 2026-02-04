import torch
import torch.nn as nn

class UNetEncoder(nn.Module):
    def __init__(self, in_channels):
        super(UNetEncoder, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv4 = nn.Conv2d(256, 512, 3, padding=1)
        self.conv5 = nn.Conv2d(512, 1024, 3, padding=1)
        
        self.pool = nn.MaxPool2d(2, 2)
        
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(256)
        self.bn4 = nn.BatchNorm2d(512)
        self.bn5 = nn.BatchNorm2d(1024)
        
        self.dropout = nn.Dropout2d(0.2)

    def forward(self, x):
        x1 = torch.relu(self.bn1(self.conv1(x)))
        x = self.pool(x1)
        x = self.dropout(x)
        
        x2 = torch.relu(self.bn2(self.conv2(x)))
        x = self.pool(x2)
        x = self.dropout(x)
        
        x3 = torch.relu(self.bn3(self.conv3(x)))
        x = self.pool(x3)
        x = self.dropout(x)
        
        x4 = torch.relu(self.bn4(self.conv4(x)))
        x = self.pool(x4)
        x = self.dropout(x)
        
        x5 = torch.relu(self.bn5(self.conv5(x)))
        return x1, x2, x3, x4, x5

class UNetDecoder(nn.Module):
    def __init__(self, out_channels):
        super(UNetDecoder, self).__init__()
        self.upconv5 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.upconv4 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.upconv3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.upconv2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.upconv1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        
        self.conv5 = nn.Conv2d(1024, 512, 3, padding=1)
        self.conv4 = nn.Conv2d(512, 256, 3, padding=1)
        self.conv3 = nn.Conv2d(256, 128, 3, padding=1)
        self.conv2 = nn.Conv2d(128, 64, 3, padding=1)
        self.conv1 = nn.Conv2d(32, out_channels, 3, padding=1)
        
        self.bn5 = nn.BatchNorm2d(512)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn2 = nn.BatchNorm2d(64)
        
        self.dropout = nn.Dropout2d(0.2)

    def forward(self, x1, x2, x3, x4, x5):
        x = self.upconv5(x5)
        x = torch.cat([x, x4], dim=1)
        x = torch.relu(self.bn5(self.conv5(x)))
        x = self.dropout(x)
        
        x = self.upconv4(x)
        x = torch.cat([x, x3], dim=1)
        x = torch.relu(self.bn4(self.conv4(x)))
        x = self.dropout(x)
        
        x = self.upconv3(x)
        x = torch.cat([x, x2], dim=1)
        x = torch.relu(self.bn3(self.conv3(x)))
        x = self.dropout(x)
        
        x = self.upconv2(x)
        x = torch.cat([x, x1], dim=1)
        x = torch.relu(self.bn2(self.conv2(x)))
        x = self.dropout(x)
        
        x = self.upconv1(x)
        x = self.conv1(x)
        return torch.sigmoid(x)

class UNetSteganoModel(nn.Module):
    def __init__(self):
        super(UNetSteganoModel, self).__init__()
        self.encoder = UNetEncoder(6)
        self.decoder = UNetDecoder(3)
        
    def forward(self, cover, secret):
        original_size = cover.size()[2:]
        x = torch.cat((cover, secret), dim=1)
        x1, x2, x3, x4, x5 = self.encoder(x)
        encoded = self.decoder(x1, x2, x3, x4, x5)
        
        if encoded.size() != cover.size():
            encoded = nn.functional.interpolate(encoded, size=original_size, mode='bilinear', align_corners=False)
        
        stego = cover + encoded
        stego = torch.clamp(stego, 0, 1)
        
        stego_features = self.encoder(torch.cat((stego, stego), dim=1))
        revealed = self.decoder(*stego_features)
        
        if revealed.size() != secret.size():
            revealed = nn.functional.interpolate(revealed, size=original_size, mode='bilinear', align_corners=False)
        
        return stego, revealed 