from PIL import Image
import os
from skimage.transform import resize
import numpy as np
import pandas as pd
from google.cloud import storage
import io
def open_files(category=1,train=True):
    """
    Open all transformed images as PIL images
    return => a list of PIL images
    """
    images_cat=[]
    filenames_cat=[]
    filenames=[]
    images=[]
    X_test=[]
    Y_test=[]


    if os.environ.get("FILE_TARGET") == "gcs":

        client = storage.Client()
        bucket_transfo = client.bucket("data-transfo")

        # Téléchargement du CSV
        blob_csv = bucket_transfo.blob("train_wkt_v4.csv")
        csv_data = blob_csv.download_as_bytes()
        trainwkd = pd.read_csv(io.BytesIO(csv_data))
        names = trainwkd['ImageId'].drop_duplicates()

        # Liste des fichiers GeoJSON
        geojson_prefix = f"three_band_geo_proc/Class_{category}/"
        blobs_geo = list(bucket_transfo.list_blobs(prefix=geojson_prefix))

        # Liste des fichiers d'images
        img_prefix = "three_band_preproc/"
        blobs_img = list(bucket_transfo.list_blobs(prefix=img_prefix))

        if train:
            for blob_geo in blobs_geo:
                filename = blob_geo.name.split("/")[-1]
                print(filename)
                geojson_data = blob_geo.download_as_bytes()
                geojson_image = Image.open(io.BytesIO(geojson_data))
                if  "6100_2_2" in filename:
                    Y_test.append(geojson_image)
                else:
                    images_cat.append(geojson_image)
                    filenames_cat.append(filename.split('.')[0][:-6])

            cats = filenames_cat.copy()
            while len(cats) > 0:
                for blob_img in blobs_img:
                    filename = blob_img.name.split("/")[-1]
                    if len(cats) == 0:
                        break
                    if "6100_2_2" in filename:
                        img_data = blob_img.download_as_bytes()
                        img = Image.open(io.BytesIO(img_data))
                        X_test.append(img)
                    if filename.split('.')[0] == cats[0]:
                        cats.pop(0)
                        img_data = blob_img.download_as_bytes()
                        img = Image.open(io.BytesIO(img_data))
                        images.append(img)
                        filenames.append(filename.split('.')[0])


    else:
        trainwkd=pd.read_csv('Data/raw_data/train_wkt_v4.csv')
        names=trainwkd['ImageId'].drop_duplicates()
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


    return images_cat,images, X_test, Y_test

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

def data_augmentation(format_crop=3335,resize_params=512,train=True,category=1):
    train_X,test_X,train_Y, test_Y = cropped_resized_images(format_crop=format_crop,resize_params=resize_params,train=train,category=category)
    X_train_aug = []
    train_Y_aug = []

    def rotate_dataset(ds):
        ds_aug = []
        for image in ds:
            for i in range(4):
                ds_aug.append(np.rot90(image, k=i))
        ds_array = np.array(ds_aug)
        return ds_array


    X_train_aug = rotate_dataset(train_X)
    train_Y_aug = rotate_dataset(train_Y)


    return X_train_aug, test_X, train_Y_aug, test_Y

def cropped_resized_images(format_crop=3335,resize_params=512,train=True,category=1):
    """
    crops and resizes all images and put them in a list
    return => a df of plt arrays and image name
    """
    image_cat, images, X_test,Y_test=open_files(train=train, category=category)
    processed_image_X = None  # Initialisation
    processed_image_Y = None  # Initialisation



    if len(image_cat) == 0:
        processed_image=[resize(np.array(cropping(X=image,format_crop=format_crop))/255,(resize_params,resize_params,3)) for image in images]

    else:
        train_X= np.array([resize(np.array(cropping(X=image,format_crop=format_crop))/255,(resize_params,resize_params,3)) for image in images])
        test_X= np.array([resize(np.array(cropping(X=image,format_crop=format_crop))/255,(resize_params,resize_params,3)) for image in X_test])

        train_Y=np.array([binary(resize(np.array(cropping(X=image,format_crop=format_crop))/255,(resize_params,resize_params,3))) for image in image_cat])
        test_Y=  np.array([binary(resize(np.array(cropping(X=image,format_crop=format_crop))/255,(resize_params,resize_params,3))) for image in Y_test])
    return train_X,test_X,train_Y, test_Y


if __name__=='__main__':
    open_files(category=1,train=True)
