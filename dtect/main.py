"""
Main file : Project-D-tect

"""
import os
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import pandas as pd
from dtect.Data_preparation.preprocessing import cropped_resized_images
from dtect.Model.UNet_v0 import UNet



def train_model(model, optimizer, criterion, num_epochs=10, image_size=128,category=1,train=True):
    X,Y=cropped_resized_images(train=train, category=category, resize_params=image_size)


    for epoch in range(num_epochs):



        X_tensor = torch.from_numpy(X.reshape(-1, 3, image_size, image_size).astype(np.float32))
        Y_tensor = torch.from_numpy(Y.reshape(-1, 1, image_size, image_size).astype(np.float32)).float()

        outputs = model(X_tensor)
        loss = criterion(outputs, Y_tensor)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

    print("Model training complete")

def main(category, image_size,lr=0.001):



    model = UNet()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()

    train_model(model, optimizer, criterion, num_epochs=10)

    print("All steps completed successfully")

if __name__ == "__main__":
    main(1,128)
