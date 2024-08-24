import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from dtect.Data_preparation.preprocessing import cropped_resized_images
import matplotlib.pyplot as plt
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels),
        )
    def forward(self, x):
        return self.conv(x)
class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features=[64, 128, 256,512]):
        super(UNet, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # Downward path (Encoder)
        self.down1 = DoubleConv(in_channels, features[0])
        self.down2 = DoubleConv(features[0], features[1])
        self.down3 = DoubleConv(features[1], features[2])
        # Bottleneck
        self.bottleneck = DoubleConv(features[2], features[2] * 2)
        # Upward path (Decoder)
        self.up1 = nn.ConvTranspose2d(features[2] * 2, features[2], kernel_size=2, stride=2)
        self.up_conv1 = DoubleConv(features[2] * 2, features[2])
        self.up2 = nn.ConvTranspose2d(features[2], features[1], kernel_size=2, stride=2)
        self.up_conv2 = DoubleConv(features[1] * 2, features[1])
        self.up3 = nn.ConvTranspose2d(features[1], features[0], kernel_size=2, stride=2)
        self.up_conv3 = DoubleConv(features[0] * 2, features[0])
        # Final convolution
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)
    def forward(self, x):
        # Downward path
        skip_connections = []
        x1 = self.down1(x)
        skip_connections.append(x1)
        x = self.pool(x1)
        x2 = self.down2(x)
        skip_connections.append(x2)
        x = self.pool(x2)
        x3 = self.down3(x)
        skip_connections.append(x3)
        x = self.pool(x3)
        # Bottleneck
        x = self.bottleneck(x)
        # Upward path
        x = self.up1(x)
        x = torch.cat((x, skip_connections[2]), dim=1)
        x = self.up_conv1(x)
        x = self.up2(x)
        x = torch.cat((x, skip_connections[1]), dim=1)
        x = self.up_conv2(x)
        x = self.up3(x)
        x = torch.cat((x, skip_connections[0]), dim=1)
        x = self.up_conv3(x)
        return torch.sigmoid(self.final_conv(x))
    def prediction(self, X):
        with torch.no_grad():
            return self.forward(torch.from_numpy(X.reshape(1, 3, 128, 128).astype(np.float32))).detach().numpy().reshape(128, 128)
if __name__ == '__main__':
    # Initialize the model
    model = UNet()
    # Optimizer & loss function (Adam & Binary-cross entropy)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCELoss()
    # Example with 1 image
    images = cropped_resized_images(train=True, category=2, resize_params=128)
    x = torch.from_numpy(images[0].reshape(-1, 3, 128, 128).astype(np.float32))
    y = torch.from_numpy(images[1].reshape(-1, 1, 128, 128).astype(np.float32)).float()
    # Forward pass
    for epoch in range(30):
        model.train()
        output = model(x)
        # Calculate the loss
        loss = criterion(output, y)
        # Backward pass & optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f'Loss: {loss.item()}')
    # Prediction (X_test)
    X_test = images[0][5]
    model.eval()  # Evaluation mode to disable dropout and batch normalization
    pred = model.prediction(X_test)
    # Convert the list of predictions into a NumPy array
    predictions = np.array(pred)
    print(f'Predictions shape: {predictions.shape}')
    plt.imshow(predictions, cmap='gray')
