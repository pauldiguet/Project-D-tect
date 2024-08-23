import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import numpy as np
from dtect.Data_preparation.preprocessing import cropped_resized_images

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        # Partie descendante (Encoder)
        self.enc1 = self.conv_block(3, 64, kernel_size=3, padding='same')
        self.enc2 = self.conv_block(64, 128, kernel_size=3, padding='same')
        self.enc3 = self.conv_block(128, 256, kernel_size=3, padding='same')
        self.enc4 = self.conv_block(256, 512, kernel_size=3, padding='same')
        self.bottleneck = self.conv_block(512, 1024, kernel_size=3, padding='same')

        # Partie ascendante (Decoder)
        self.upsamp4 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.downconv4 = nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=2, stride=1, padding='same')
        self.dec4 = self.conv_block(1024, 512, kernel_size=3, padding='same')

        self.upsamp3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.downconv3 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=2, stride=1, padding='same')
        self.dec3 = self.conv_block(512, 256, kernel_size=3, padding='same')

        self.upsamp2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.downconv2 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=2, stride=1, padding='same')
        self.dec2 = self.conv_block(256, 128, kernel_size=3, padding='same')

        self.upsamp1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.downconv1 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=2, stride=1, padding='same')
        self.dec1 = self.conv_block(128, 64, kernel_size=3, padding='same')

        # Couches de sortie
        self.conv_out = nn.Conv2d(64, 1, kernel_size=1)
        self.sigm = nn.Sigmoid()

    def conv_block(self, in_channels, out_channels, kernel_size, padding):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)
        shapes.append(enc1.shape)
        enc2 = self.enc2(F.max_pool2d(enc1, 2))
        shapes.append(enc2.shape)
        enc3 = self.enc3(F.max_pool2d(enc2, 2))
        shapes.append(enc3.shape)
        enc4 = self.enc4(F.max_pool2d(enc3, 2))
        shapes.append(enc4.shape)

        bottleneck = self.bottleneck(F.max_pool2d(enc4, 2))
        shapes.append(bottleneck.shape)

        # Decoder with skip connections
        dec4 = self.upsamp4(bottleneck)
        dec4 = self.downconv4(dec4)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.dec4(dec4)
        shapes.append(dec4.shape)

        dec3 = self.upsamp3(dec4)
        dec3 = self.downconv3(dec3)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.dec3(dec3)
        shapes.append(dec3.shape)

        dec2 = self.upsamp2(dec3)
        dec2 = self.downconv2(dec2)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.dec2(dec2)
        shapes.append(dec2.shape)

        dec1 = self.upsamp1(dec2)
        dec1 = self.downconv1(dec1)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.dec1(dec1)
        shapes.append(dec1.shape)

        out = self.conv_out(dec1)
        pred = self.sigm(out)

        return pred


if __name__ == '__main__':
    # Initialiser le modèle
    shapes = []
    model = UNet()

    # optimizer & loss function (Adam & Binary-cross entropy)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCELoss()

    # Exemple avec 1 image
    x = torch.from_numpy(cropped_resized_images(train=True, category=2, resize_params=128)[0].reshape(15, 3, 128, 128).astype(np.float32))
    y = torch.from_numpy(cropped_resized_images(train=True, category=2, resize_params=128)[1].reshape(15, 1, 128, 128).astype(np.float32)).float()
    # Forward pass
    output = model(x)

    # Calcul de la perte
    loss = criterion(output, y)

    # Backward pass & optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f'Loss: {loss.item()}')
    print(pd.DataFrame(shapes))
