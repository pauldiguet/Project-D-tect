import os
import time
import torch

from dtect.Model.UNet_v0 import UNet
from google.cloud import storage

def save_model(model = None) -> None:
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    client = storage.Client()
    bucket = client.bucket(os.environ.get("BUCKET_TRANSFO"))
    blob = bucket.blob(f"data-results/{timestamp}.h5")
    torch.save(model.state_dict(), blob)

    print("✅ Model saved to GCS")

    return None


def load_model() -> torch.Model:
    client = storage.Client()
    bucket = client.bucket(os.environ.get("BUCKET_TRANSFO"))
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
        print(f"\n❌ No model found in GCS bucket {os.environ.get("BUCKET_TRANSFO")}")

        return None
