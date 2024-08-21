import geopandas as gpd
import imageio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from os.path import exists

def get_names():

    """
    Get's the names of the different images:

    return => data series
    """

    train_wkt=pd.read_csv('Data/raw_data/train_wkt_v4.csv')
    return train_wkt['ImageId'].drop_duplicates()

def grid_values():
    """
    Get's the grid values for each image:
    return => data frame
    """
    grid= pd.read_csv('Data/raw_data/grid_sizes.csv')
    return grid.set_index('Unnamed: 0', inplace= True)

def geojsons(folder_path):

    """
    Open and reads all geojsons for each folder
    input => path (str)
    return => filename: list of str
               geojson: list of dataframes

    """
    file_name=[]
    geojson=[]
    for filename in os.listdir(folder_path):

        file_path = os.path.join(folder_path, filename)

        img = gpd.read_file(file_path)

        file_name.append(filename)
        geojson.append(img)
    return file_name, geojson

def categorize_files(file_name):
    """
    Categories geojson using their names into 8 categories, ignores Grid
    input => filename: str
    return => list of categories: str
    """
    categories = []
    for name in file_name:
        if "Grid" not in name:
            categories.append(name.split('_')[0][2])
    return categories

def save_mask():
    """
    Saves the geojson mask as jpg.

    Specification to jpg name:

    imagename cat_number  repetition num (optional) .jpg

    6010_2_2 cat_1 1 .jpg --> 6010_2_2_cat_1_1.jpg

    repetition number: if an image has recurring category it will add a repetition number to differenciate the images

    """
    # Getting values
    grid=grid_values()
    names=get_names()


    for name in names:
        #opens geojson and categorizes the names
        file_name, geojson=geojsons(f'Data/raw_data/train_geojson_v3/{name}')
        categories=categorize_files(file_name=file_name)
        u=0
        k=1
        repetition_tracker = {category: 1 for category in set(categories)}   #this makes us able later to name according to repetision

        for i in range(len(file_name)):
            if 'Grid' not in file_name[i]:
                file_exists = exists(f'Data/processed_data/three_band_geo_proc/Class_{categories[u]}/{name}_cat_{categories[u]}.jpg')
                category = categories[u]
                if file_exists:
                    fig, ax = plt.subplots(figsize=(11.16, 11.3))
                    geojson[i].plot()
                    plt.xlim(0,grid.loc[f'{name}']['Xmax'])
                    plt.ylim(grid.loc[f'{name}']['Ymin'],0)
                    plt.axis("off")
                    plt.savefig(f'Data/processed_data/three_band_geo_proc/Class_{categories[u]}/{name}_cat_{categories[u]}_{repetition_tracker[category]}.jpg', bbox_inches='tight', pad_inches=0, dpi=300)
                    plt.close(fig)
                    repetition_tracker[category] += 1
                    u+=1

                else:
                    fig, ax = plt.subplots(figsize=(11.16, 11.3))
                    geojson[i].plot()
                    plt.xlim(0,grid.loc[f'{name}']['Xmax'])
                    plt.ylim(grid.loc[f'{name}']['Ymin'],0)
                    plt.axis("off")
                    plt.savefig(f'Data/processed_data/three_band_geo_proc/Class_{categories[u]}/{name}_cat_{categories[u]}.jpg', bbox_inches='tight', pad_inches=0, dpi=300)
                    plt.close(fig)
                    u+=1

if __name__=="__main__":
    save_mask()
