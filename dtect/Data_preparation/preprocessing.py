import matplotlib.pyplot as plt
from PIL import Image
import os
from skimage.transform import resize
import numpy as np
import pandas as pd

def open_files(folder_name, train=False):
    """
    Open all transformed images as PIL images
    return => a list of PIL images
    """
    images=[]
    filenames=[]

    train_wkt=pd.read_csv('Data/raw_data/train_wkt_v4.csv')


    if train==True:
        for filename in os.listdir(f'Data/processed_data/{folder_name}'):
                file_path = os.path.join(f'Data/processed_data/{folder_name}', filename)
                if filename.split('.')[0] in train_wkt['ImageId'].drop_duplicates():
                    img = Image.open(file_path)
                    images.append(img)
                    filenames.append(filename)


    if folder_name == 'three_band_geo_proc':
        for category in os.listdir(f'Data/processed_data/{folder_name}'):
            for filename in category:
                file_path = os.path.join(f'Data/processed_data/{folder_name}/{category}', filename)
                img = Image.open(file_path)
                images.append(img)
                filenames.append(filename)


    else:
        for filename in os.listdir(f'Data/processed_data/{folder_name}'):
                file_path = os.path.join(f'Data/processed_data/{folder_name}', filename)
                if file_path!= f"Data/processed_data/{folder_name}/.DS_Store" and file_path!= f"Data/processed_data/{folder_name}/.gitkeep":
                    img = Image.open(file_path)
                    images.append(img)
                    filenames.append(filename)


    return images, filenames

def cropping(X,format_crop):
    """
    Crops image to desired shape of (0, 0, 3335, 3335)
    input => PIL image
    return => cropped PIL image
    """
    format_cropped = (0, 0, format_crop, format_crop) # cropping rule basé sur les dimensions les plus réduites du dataset
    return X.crop(format_cropped)


def cropped_resized_images(folder_name,format_crop,resize_params,train=False):
    """
    crops and resizes all images and put them in a list
    return => a list of plt arrays
    """
    images, filenames =open_files(folder_name=folder_name,train=train)

    processed_image={}
    i=0
    for image in images:
        processed_image[filenames[i].split('.')[0]]=resize(np.array(cropping(image, format_crop)),(resize_params,resize_params,3))
        i+=1

    return processed_image


if __name__ == '__main__':
    cropped_resized_images(folder_name='three_band_preproc',format_crop=3335,resize_params=544,train=True)
