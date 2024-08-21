import matplotlib.pyplot as plt
from PIL import Image
import os
from skimage.transform import resize
import numpy as np

def open_files(folder_name):
    """
    Open all transformed images as PIL images
    return => a list of PIL images
    """
    images=[]

    if folder_name == 'three_band_geo_proc':
        for category in os.listdir(f'Data/processed_data/{folder_name}'):
            for filename in category:
                file_path = os.path.join(f'Data/processed_data/{folder_name}/{category}', filename)
                img = Image.open(file_path)
                images.append(img)

    else:
        for filename in os.listdir(f'Data/processed_data/{folder_name}'):
                file_path = os.path.join(f'Data/processed_data/{folder_name}', filename)
                if file_path!= f"Data/processed_data/{folder_name}/.DS_Store" and file_path!= f"Data/processed_data/{folder_name}/.gitkeep":
                    img = Image.open(file_path)
                    images.append(img)

    return images

def cropping(X,format_crop):
    """
    Crops image to desired shape of (0, 0, 3335, 3335)
    input => PIL image
    return => cropped PIL image
    """
    format_cropped = (0, 0, format_crop, format_crop) # cropping rule basé sur les dimensions les plus réduites du dataset
    return X.crop(format_cropped)


def cropped_resized_images(folder_name,format_crop,resize_params):
    """
    crops and resizes all images and put them in a list
    return => a list of plt arrays
    """
    images=open_files(folder_name)

    processed_image=[]
    for image in images:
        processed_image.append(resize(np.array(cropping(image, format_crop)),(resize_params,resize_params,3)))

    return processed_image
