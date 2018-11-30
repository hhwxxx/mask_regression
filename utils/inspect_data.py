from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import shutil

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw


def draw_outline(image, coordinates):
    """Draw mask outline.
    
    Args:
        image: Instance os class Image.
        coordinates: mask coordinates (top_left, top_right, bottom_right, bottom_left).
    
    Returns:
        Class Image instance with mask outline.
    """


def main(csv_file, save_dir):
    csv = pd.read_csv(csv_file)
    for i in range(10):
        filename = csv.iloc[i]['filename']
        filename = os.path.join('/root/mask', filename[2:])
        
        top_left_height = csv.iloc[i]['top_left_height']
        top_left_width = csv.iloc[i]['top_left_width']
        top_right_height = csv.iloc[i]['top_right_height']
        top_right_width = csv.iloc[i]['top_right_width']
        bottom_left_height = csv.iloc[i]['bottom_left_height']
        bottom_left_width = csv.iloc[i]['bottom_left_width']
        bottom_right_height = csv.iloc[i]['bottom_right_height']
        bottom_right_width = csv.iloc[i]['bottom_right_width']
        
        top_left = (top_left_width, top_left_height)
        top_right = (top_right_width, top_right_height)
        bottom_left = (bottom_left_width, bottom_left_height)
        bottom_right = (bottom_right_width, bottom_right_height)
        
        coordinates = [top_left, top_right, bottom_right, bottom_left]

	    img = Image.open(filename)

        img = draw_outline(img, coordinates)
        
        save_img = os.path.join(save_dir, os.path.basename(filename))
        img.save(save_img, format='PNG')


if __name__ == '__main__':
    csv_file = '/root/mask/mask_data/quadrilateral/train.csv'
    save_dir = '/root/mask/utils/test/'
    
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    os.mkdir(save_dir)
 
    main(csv_file, save_dir)
