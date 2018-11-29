from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import os
import ast

import pandas as pd
import numpy as np

CSV_PATH = '/home/hhw/work/hikvision/utils/test'
SAVE_PATH = '/home/hhw/work/hikvision/utils/test'

CSV_COLUMNS = ['filename', 'top_left_height', 'top_left_width', 
               'top_right_height', 'top_right_width',
               'bottom_left_height', 'bottom_left_width', 
               'bottom_right_height', 'bottom_right_width']


def main(dataset_split):
    """Modify csv file.

    Args:
        dataset_split: string representing dataset split.
    """
    df = pd.DataFrame(columns=CSV_COLUMNS)
    csv = pd.read_csv(os.path.join(CSV_PATH, dataset_split + '.csv'))
    for i in range(len(csv)):
        filename = csv.iloc[i]['filename']

        top_left_height = ast.literal_eval(
            str(csv.iloc[i]['top_left_height']))
        top_left_width = ast.literal_eval(
            str(csv.iloc[i]['top_left_width']))
        top_right_height = ast.literal_eval(
            str(csv.iloc[i]['top_right_height']))
        top_right_width = ast.literal_eval(
            str(csv.iloc[i]['top_right_width']))
        bottom_left_height = ast.literal_eval(
            str(csv.iloc[i]['bottom_left_height']))
        bottom_left_width = ast.literal_eval(
            str(csv.iloc[i]['bottom_left_width']))
        bottom_right_height = ast.literal_eval(
            str(csv.iloc[i]['bottom_right_height']))
        bottom_right_width = ast.literal_eval(
            str(csv.iloc[i]['bottom_right_width']))
        
        coordinates = [top_left_height, top_left_width, 
                       top_right_height, top_right_width,
                       bottom_left_height, bottom_left_width,
                       bottom_right_height, bottom_right_width]

        if len(top_left_height) > 1:
            height_list = list(zip(top_left_height, top_right_height))
            if min(height_list[0]) > min(height_list[1]):
                coordinates = [x[::-1] for x in coordinates]
        
        df = df.append(
            pd.DataFrame(
                [[filename] + coordinates], 
                columns=CSV_COLUMNS), ignore_index=True)
    
    df.to_csv(os.path.join(SAVE_PATH, dataset_split + '.csv'))


if __name__ == '__main__':
    dataset = ['train', 'test', 'val']
    if not os.path.exists(SAVE_PATH):
        os.makedirs(SAVE_PATH)
    for dataset_split in dataset:
        main(dataset_split)
        print('Finish processing ' + dataset_split)