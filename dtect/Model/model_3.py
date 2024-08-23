import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from dtect.Data_preparation.preprocessing import cropped_resized_images
import numpy as np
import matplotlib.pyplot as plt

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features=[64, 128, 256]):
        super(UNet, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # Downsampling path
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature
        # Upsampling path
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(
                    feature * 2, feature, kernel_size=2, stride=2
                )
            )
            self.ups.append(DoubleConv(feature * 2, feature))
        # Bottleneck
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)
    def forward(self, x):
        skip_connections = []
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)
        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx // 2]
            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx + 1](concat_skip)
        return torch.sigmoid(self.final_conv(x))
    def prediction(self,X):
        with torch.no_grad():
            return self.forward(torch.from_numpy(X.reshape(1,3, 128, 128).astype(np.float32))).detach().numpy().reshape(128,128)

if __name__ == '__main__':
    # Initialiser le modèle
    model = UNet()
    # optimizer & loss function (Adam & Binary-cross entropy)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCELoss()
    # Exemple avec 1 image
    X,Y = cropped_resized_images(train=True, category=2, resize_params=128)
    x = torch.from_numpy(X.reshape(-1, 3, 128, 128).astype(np.float32))
    y = torch.from_numpy(Y.reshape(-1, 1, 128, 128).astype(np.float32)).float()
    # Forward pass
    for epoch in range(500):
        model.train()
        output = model(x)
    # Calcul de la loss
        loss = criterion(output, y)
    # Backward pass & optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f'Loss: {loss.item()}')
    # Prediction (X_test)
    X_test = X[5]
    model.eval()  # Mode évaluation pour désactiver le dropout et la normalisation par batch
    pred = model.prediction(X_test)
    # Convertir la liste de prédictions en un tableau NumPy
    predictions = np.array(pred)
    print(f'Predictions shape: {predictions.shape}')
    plt.imshow(predictions, cmap='gray')
