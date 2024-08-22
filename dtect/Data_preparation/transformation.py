import geopandas as gpd
import glob
import imageio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from os.path import exists


def grid_values():
    """
    Get's the grid values for each image:
    return => data frame
    """
    grid = pd.read_csv('./Data/raw_data/grid_sizes.csv')
    return grid.set_index('Unnamed: 0', inplace = False)

def pixel_values():
    """
    Get's the dimensions for each image:
    return => data frame
    """
    grid = pd.read_csv('./Data/dimensions.csv')
    return grid.set_index('Unnamed: 0', inplace = False)


def export_geojson(geojson, image_id, category, grid, dimensions):

    # plt.figure(figsize=(int(dimensions[0]/200),int(dimensions[1]/200)), dpi=100)
    pd.concat(geojson).plot(figsize=(dimensions[0],dimensions[1]))
    plt.xlim(0,grid.loc[f'{image_id}']['Xmax'])
    plt.ylim(grid.loc[f'{image_id}']['Ymin'],0)
    plt.axis("off")

    # plt.savefig(f'./Data/processed_data/three_band_geo_proc/Class_{categories[u]}/{name}_cat_{categories[u]}.jpg', bbox_inches='tight', pad_inches=0, dpi=300)
    plt.savefig(f'./Data/processed_data/three_band_geo_proc/Class_{category}/{image_id}_cat_{category}.jpg',
        bbox_inches='tight',
        pad_inches=0,
        dpi=1.3091750192752505)
    print(f'Saved : {image_id}_cat_{category}.jpg', end = ' - ')
    print(f'With resolution : {dimensions[0],dimensions[1]}')
    # plt.show()
    plt.close()
    pass


def transform_geojson():
    """
    Saves the geojson mask as jpg.

    Specification to jpg name:

    imagename cat_number  repetition num (optional) .jpg

    6010_2_2 cat_1 1 .jpg --> 6010_2_2_cat_1_1.jpg

    repetition number: if an image has recurring category it will add a repetition number to differenciate the images

    """
    # Getting values
    grid = grid_values()
    pixels = pixel_values()

    gjson_dirs = glob.glob('./Data/raw_data/train_geojson_v3/*')

    for image_id in [n.split('/')[-1] for n in gjson_dirs]:

        print('Extracting geojson from image : ',image_id)

        gjson_files = glob.glob(f'./Data/raw_data/train_geojson_v3/{image_id}/*')
        gjson_files = sorted([n.split('/')[-1] for n in gjson_files])
        categories = [int(gjson_file[2]) for gjson_file in gjson_files[:-1]]

        geojson = []

        for index in range(len(categories)):

            geojson.append(gpd.read_file(f'./Data/raw_data/train_geojson_v3/{image_id}/{gjson_files[index]}'))

            if index == len(categories)-1:
                export_geojson(geojson, image_id, categories[index], grid, pixels.loc[image_id].to_list())

            else:
                if categories[index] != categories[index+1]:
                    export_geojson(geojson, image_id, categories[index], grid, pixels.loc[image_id].to_list())
                    geojson = []

if __name__=="__main__":
    transform_json()
