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
    blob = bucket.blob(f"data-results/{timestamp}.h5")

    # Télécharger le fichier local vers GCS
    blob.upload_from_filename(local_path)

    # Supprimer le fichier temporaire local pour économiser de l'espace
    os.remove(local_path)

    print("✅ Model saved on GCS")

    return None

def save_fig_pred(epoch, image_size, fig=None) -> None:

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    plt.imshow(fig, cmap='gray')

    plt.savefig(f'plot_results/plot{epoch}_{image_size}.png')
    plt.close()

    # Initialiser le client GCS et spécifier le bucket
    client = storage.Client()
    bucket = client.bucket("data-transfo")

    # Créer un blob pour le fichier dans le bucket
    blob = bucket.blob(f"data-transfo/data-results/plot_results/plot{epoch}_{image_size}_{timestamp}")

    # Télécharger le fichier local vers GCS
    blob.upload_from_filename(f'plot_results/plot{epoch}_{image_size}.png')

    # Supprimer le fichier temporaire local pour économiser de l'espace
    os.remove(f'plot_results/plot{epoch}_{image_size}.png')

    print("✅ Plot saved on GCS")

    return None


def load_model():
    client = storage.Client()
    bucket = client.bucket("data-transfo")
    blobs = bucket.blob(f"data-results/*")

    try:
        latest_blob = max(blobs, key=lambda x: x.updated)
        latest_model_path_to_save = os.path.join(".", latest_blob.name)
        latest_blob.download_to_filename(latest_model_path_to_save)
        model = UNet()
        model.load_state_dict(torch.load(latest_model_path_to_save))

        print("✅ Latest model downloaded from cloud storage")

        return model
    except:
        print(f"\n❌ No model found in GCS bucket {'data-transfo'}")

        return None
