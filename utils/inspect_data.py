from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import ast
import shutil

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

NUM_MAP = {
    'train': 5000,
    'val': 1500,
    'test': 2000,
}

def main(data_dir, dataset_split, save_dir):
    csv_file = os.path.join(data_dir, dataset_split + '.csv')
    csv = pd.read_csv(csv_file)
    for i in range(NUM_MAP[dataset_split]):
        filename = csv.iloc[i]['filename']
        filename = os.path.join('/data/mask', filename[3:])
        image = Image.open(filename)
        draw = ImageDraw.Draw(image)
        
        top_left_height = ast.literal_eval(csv.iloc[i]['top_left_height'])
        top_left_width = ast.literal_eval(csv.iloc[i]['top_left_width'])
        top_right_height = ast.literal_eval(csv.iloc[i]['top_right_height'])
        top_right_width = ast.literal_eval(csv.iloc[i]['top_right_width'])
        bottom_left_height = ast.literal_eval(csv.iloc[i]['bottom_left_height'])
        bottom_left_width = ast.literal_eval(csv.iloc[i]['bottom_left_width'])
        bottom_right_height = ast.literal_eval(csv.iloc[i]['bottom_right_height'])
        bottom_right_width = ast.literal_eval(csv.iloc[i]['bottom_right_width'])
        
        fill_color = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
        for j in range(len(top_left_height)):
            top_left = (top_left_width[j], top_left_height[j])
            top_right = (top_right_width[j], top_right_height[j])
            bottom_left = (bottom_left_width[j], bottom_left_height[j])
            bottom_right = (bottom_right_width[j], bottom_right_height[j])
            coordinates = [top_left, top_right, bottom_right, bottom_left]

            draw.polygon(coordinates, fill=fill_color[j])

        save_name = os.path.join(save_dir, os.path.basename(filename))
        image.save(save_name, format='PNG')

        # print('processing image:', save_name)


if __name__ == '__main__':
    dataset = 'quadrilateral_2'
    data_dir = '/data/mask/mask_data/' + dataset
    save_base_path = '/data/mask/inspect_data/' + dataset
    
    print('Painting dataset:', dataset) 
    dataset = ['train', 'val', 'test']
    extra = 'w_order'
    for dataset_split in dataset:
        save_dir = os.path.join(save_base_path, extra, dataset_split)
        if os.path.exists(save_dir):
            shutil.rmtree(save_dir)
        os.makedirs(save_dir)
        print('*' * 50)
        print('Start processing dataset split:', dataset_split) 
        main(data_dir, dataset_split, save_dir)
        print('Finished processing dataset split:', dataset_split) 
        print('*' * 50)
