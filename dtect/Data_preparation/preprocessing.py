import matplotlib.pyplot as plt
from PIL import Image
import os
from skimage.transform import resize
import numpy as np

def open_files():
    """
    Open all transformed images as PIL images
    return => a list of PIL images
    """
    images=[]
    filenames=[]
    for filename in os.listdir('Data/processed_data/three_band_transformed'):
            file_path = os.path.join('Data/processed_data/three_band_transformed', filename)
            if file_path!= "Data/processed_data/three_band_transformed/.DS_Store" and file_path!= "Data/processed_data/three_band_transformed/.gitkeep":
                img = Image.open(file_path)
                images.append(img)
                filenames.append(filename)
    return images, filenames

def cropping(X):
    """
    Crops image to desired shape of (0, 0, 3335, 3335)
    input => PIL image
    return => cropped PIL image
    """
    format_crop = (0, 0, 3335, 3335) # cropping rule basé sur les dimensions les plus réduites du dataset
    return X.crop(format_crop)

def resized(X):
    return resize(np.array(X), (544, 544, 3))

def cropped_resized_images():
    """
    crops and resizes all images and put them in a list
    return => a list of plt arrays
    """
    images, filenames =open_files()

    processed_image=[]
    for image in images:
        processed_image.append(resized(cropping(image)))

    return processed_image ,filenames


def save_images():
    """
    saves all images in three_band_proc
    """
    processed_image, filenames = cropped_resized_images()
    i=0
    for image in processed_image:
        file_path=f'../Data/processed_data/three_band_proc/{filenames[i]}'
        plt.imsave(file_path,image)
        i+=1

if __name__=="__main__":
    save_images()
