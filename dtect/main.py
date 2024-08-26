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
from dtect.Model.registry import save_model, save_fig_pred
from dtect.Model.model_3 import UNet
import matplotlib.pyplot as plt

def train_model(model, optimizer, criterion, num_epochs=10, image_size=128, category=1, train=True):
    X, Y = cropped_resized_images(train=train, category=category, resize_params=image_size)

    for epoch in range(num_epochs):

        X_tensor = torch.from_numpy(X.reshape(-1, 3, image_size, image_size).astype(np.float32))
        Y_tensor = torch.from_numpy(Y.reshape(-1, 1, image_size, image_size).astype(np.float32)).float()

        model.train()  # Mode entraînement
        print("passed train mod")
        outputs = model(X_tensor)
        loss = criterion(outputs, Y_tensor)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

        model.eval()  # Mode évaluation
        X_test = X[5].reshape(1, 3, image_size, image_size)  # Adapter la dimension de X_test
        X_test_tensor = torch.from_numpy(X_test.astype(np.float32))
        pred = model(X_test_tensor)
        predictions = pred.squeeze().detach().numpy()  # Retirer le tenseur et convertir en numpy
        print(f'Predictions shape: {predictions.shape}')
        save_fig_pred(epoch, image_size, predictions)

        if (epoch + 1) % 100 == 0:
            save_model(model.eval())  # Sauvegarder le modèle en mode évaluation

    save_model(model.eval())  # Sauvegarder le modèle en mode évaluation

    print("Model training complete")


def main(category=1, image_size=128, lr=0.01, epochs=250):
    model = UNet()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()

    train_model(model, optimizer, criterion, num_epochs=epochs, image_size=image_size, category=category)

    print("All steps completed successfully")

if __name__ == "__main__":
    main(8, 512, epochs=400)
