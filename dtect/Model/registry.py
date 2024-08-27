import os
import time
import torch

from dtect.Model.UNet_v0 import UNet
from google.cloud import storage
import matplotlib.pyplot as plt

def save_model(model=None) -> None:
    # Générer un horodatage
    timestamp = time.strftime("%Y%m%d-%H%M%S")

    # Chemin vers le fichier temporaire local
    local_path = f"model_results/{timestamp}.h5"

    # Sauvegarder le modèle localement
    torch.save(model.state_dict(), local_path)

    # Initialiser le client GCS et spécifier le bucket
    client = storage.Client()
    bucket = client.bucket("data-transfo")

    # Créer un blob pour le fichier dans le bucket
    blob = bucket.blob(f"data-results/model_results/{timestamp}.h5")

    # Télécharger le fichier local vers GCS
    blob.upload_from_filename(local_path)

    # Supprimer le fichier temporaire local pour économiser de l'espace
    os.remove(local_path)

    print("✅ Model saved on GCS")

    return None

def save_fig_pred(epoch, image_size, category=1,fig=None) -> None:

    timestamp = time.strftime("%Y%m%d-%H%M%S")

    local_path = f'plot_results/plot{epoch}_{image_size}_{timestamp}_{category}.png'

    plt.imshow(fig, cmap='gray')
    plt.savefig(local_path)
    plt.close()

    # Initialiser le client GCS et spécifier le bucket
    client = storage.Client()
    bucket = client.bucket("data-transfo")

    # Créer un blob pour le fichier dans le bucket
    blob = bucket.blob(f"data-results/plot_results/plot{epoch}_{image_size}_{timestamp}_{category}.png")

    # Télécharger le fichier local vers GCS
    blob.upload_from_filename(local_path)

    # Supprimer le fichier temporaire local pour économiser de l'espace
    os.remove(local_path)

    print("✅ Plot saved on GCS")

    return None


def load_model():

    def get_latest_model_blob(bucket_name, prefix):
        # Initialize the GCS client
        client = storage.Client()
        bucket = client.bucket(bucket_name)

        # List all blobs with the given prefix
        blobs = list(bucket.list_blobs(prefix=prefix))

        if not blobs:
            print(f"No blobs found with prefix '{prefix}' in bucket '{bucket_name}'")
            return None

        # Sort blobs by their updated time (creation time if no updates)
        blobs.sort(key=lambda x: x.updated, reverse=True)

        # Return the latest blob
        return blobs[0]

    timestamp = time.strftime("%Y%m%d-%H%M%S")

    latest_blob = get_latest_model_blob("data-transfo", "data-results/model_results/2024")
    latest_blob.download_to_filename(f"model_results/model_load{timestamp}.h5")
    model = torch.load("model_results")
    return model


def save_fig_Y(category=1,fig=None) -> None:

    timestamp = time.strftime("%Y%m%d-%H%M%S")

    local_path = f'plot_results/plot_Y_{category}.png'

    plt.imshow(fig, cmap='gray')
    plt.savefig(local_path)
    plt.close()

    # Initialiser le client GCS et spécifier le bucket
    client = storage.Client()
    bucket = client.bucket("data-transfo")

    # Créer un blob pour le fichier dans le bucket
    blob = bucket.blob(f"data-results/plot_results/plot_Y_{timestamp}_{category}.png")

    # Télécharger le fichier local vers GCS
    blob.upload_from_filename(local_path)

    # Supprimer le fichier temporaire local pour économiser de l'espace
    os.remove(local_path)

    print("✅ Plot of Y saved on GCS")

    return None

if __name__ == "__main__":
    load_model()
