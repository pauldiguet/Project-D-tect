import matplotlib.pyplot as plt
from PIL import Image
import os
from skimage.transform import resize
import numpy as np
import pandas as pd
from google.cloud import storage

def open_files(category=1,train=False):
    """
    Open all transformed images as PIL images
    return => a list of PIL images
    """
    images_cat=[]
    filenames_cat=[]
    filenames=[]
    images=[]
    trainwkd=pd.read_csv('Data/raw_data/train_wkt_v4.csv')
    names=trainwkd['ImageId'].drop_duplicates()

    if os.environ.get("FILE_TARGET") == "gcs":
        client = storage.Client()
        bucket_transfo = client.bucket(os.environ.get("BUCKET_TRANSFO"))
        Images = bucket_transfo.blob(f"three_band_preproc")
        geojsons = bucket_transfo.blob(f"three_band_geo_proc/Class_{category}")

        if train==True:
            for filename in os.listdir(geojsons):
                file_path = os.path.join(geojsons, filename)
                img = Image.open(file_path)
                images_cat.append(img)
                filenames_cat.append(filename.split('.')[0][:-6])

            u=0
            cats=filenames_cat.copy()
            while len(cats)>0:
                for filename in os.listdir(Images):
                        file_path = os.path.join(Images, filename)
                        if len(cats)==0:
                            break
                        if filename.split('.')[0] == cats[0]:
                            cats.pop(0)
                            img = Image.open(file_path)
                            images.append(img)
                            filenames.append(filename.split('.')[0])

    else:
        if train==True:
            for filename in os.listdir(f'Data/processed_data/three_band_geo_proc/Class_{category}'):
                file_path = os.path.join(f'Data/processed_data/three_band_geo_proc/Class_{category}', filename)
                if file_path!= f"Data/processed_data/three_band_geo_proc/Class_{category}/.DS_Store" and file_path!= f"Data/processed_data/three_band_geo_proc/Class_{category}/.gitkeep":
                    img = Image.open(file_path)
                    images_cat.append(img)
                    filenames_cat.append(filename.split('.')[0][:-6])
            u=0
            cats=filenames_cat.copy()

            while len(cats)>0:
                for filename in os.listdir(f'Data/processed_data/three_band_preproc'):
                        file_path = os.path.join(f'Data/processed_data/three_band_preproc', filename)
                        if len(cats)==0:
                            break
                        if filename.split('.')[0] == cats[0]:
                            cats.pop(0)
                            img = Image.open(file_path)
                            images.append(img)
                            filenames.append(filename.split('.')[0])


        else:
            for filename in os.listdir(f'Data/processed_data/three_band_preproc'):
                    file_path = os.path.join(f'Data/processed_data/three_band_preproc', filename)
                    if filename.split('.')[0] not in names:
                        if file_path!= f"Data/processed_data/three_band_preproc/.DS_Store" and file_path!= f"Data/processed_data/three_band_preproc/.gitkeep":
                            img = Image.open(file_path)
                            images.append(img)
                            filenames.append(filename)


    return images_cat,images

def cropping(X,format_crop):
    """
    Crops image to desired shape of (0, 0, 3335, 3335)
    input => PIL image
    return => cropped PIL image
    """
    format_cropped = (0, 0, format_crop, format_crop) # cropping rule basé sur les dimensions les plus réduites du dataset
    return X.crop(format_cropped)

def binary(rgb_array):
    """
    Converts an RGB image array to a binary image array without converting to binary.

    - numpy.ndarray: A binary image array with shape (height, width) and values 0 or 1.
    """

    r_channel = rgb_array[:,:, 0] > 0.9
    g_channel = rgb_array[:,:, 1] > 0.9
    b_channel = rgb_array[:,:,2] > 0.9

    binary_array = np.where(r_channel | g_channel | b_channel, 1, 0)

    final_array=np.expand_dims(binary_array, -1)

    return final_array

def cropped_resized_images(format_crop=3335,resize_params=544,train=False,category=1):
    """
    crops and resizes all images and put them in a list
    return => a df of plt arrays and image name
    """
    image_cat,  images  =open_files(train=train, category=category)

    if len(image_cat) == 0:
        processed_image=[resize(np.array(cropping(X=image,format_crop=format_crop))/255,(resize_params,resize_params,3)) for image in images]

    else:
        processed_image_X=np.array([resize(np.array(cropping(X=image,format_crop=format_crop))/255,(resize_params,resize_params,3)) for image in images])
        processed_image_Y=np.array([binary(resize(np.array(cropping(X=image,format_crop=format_crop))/255,(resize_params,resize_params,3))) for image in image_cat])
    return processed_image_X, processed_image_Y
