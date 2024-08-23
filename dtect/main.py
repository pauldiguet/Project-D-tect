"""
Main file : Project-D-tect

"""
import os
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import pandas as pd
from Data_preparation.preprocessing import cropped_resized_images
from dtect.Model.model_3 import UNet
import matplotlib.pyplot as plt

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

        X_test = X[5]
        model.eval()  # Mode évaluation pour désactiver le dropout et la normalisation par batch
        pred = model.prediction(X_test)
        # Convertir la liste de prédictions en un tableau NumPy
        predictions = np.array(pred)
        print(f'Predictions shape: {predictions.shape}')
        plt.show()

    print("Model training complete")

def main(category=1, image_size=512,lr=0.001):
    model = UNet()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()

    train_model(model, optimizer, criterion, num_epochs=1,image_size=image_size,category=category)

    print("All steps completed successfully")

if __name__ == "__main__":
    main(1,512)
