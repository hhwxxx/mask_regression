from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from PIL import Image, ImageDraw
import random
import os
import pandas as pd


shape = (224, 224, 3)
height = shape[0]
width = shape[1]


def main():
    # Rectangle
    save_dirs = ['./mask_data/rectangle_3channel/train', 
                 './mask_data/rectangle_3channel/val',
                 './mask_data/rectangle_3channel/test']
    csv_files = ['./mask_data/rectangle_3channel/train.csv',
                 './mask_data/rectangle_3channel/val.csv',
                 './mask_data/rectangle_3channel/test.csv']
    NUM_IMAGES = [5000, 1500, 2000]

    csv_columns = ['filename', 'top_left_height', 'top_left_width', 
                   'top_right_height', 'top_right_width',
                   'bottom_left_height', 'bottom_left_width', 
                   'bottom_right_height', 'bottom_right_width']

    for save_dir, csv_file, NUM_IMAGE in zip(save_dirs, csv_files, NUM_IMAGES):
        num_image = 0
        df = pd.DataFrame(columns=csv_columns)
        while True:
            image = np.zeros(shape, np.uint8)
            # background is (128, 128, 128)
            image.fill(128)
        
            # create mask coordinates
            top_left_height = random.randint(0, height - 1)
            top_left_width = random.randint(0, width - 1)
            top_right_height = top_left_height
            top_right_width = random.randint(top_left_width, width - 1)
            bottom_left_height = random.randint(top_left_height, height - 1)
            bottom_left_width = top_left_width
            bottom_right_height = bottom_left_height
            bottom_right_width = top_right_width
        
            assert top_right_width >= top_left_width
            assert bottom_left_height >= top_left_height
        
            if bottom_right_height - top_left_height < 30 \
                or bottom_right_width - top_left_width < 30:
                continue
        
            # Point is represented as (width, height) in PIL.
            top_left = (top_left_width, top_left_height)
            top_right = (top_right_width, top_right_height)
            bottom_left = (bottom_left_width, bottom_left_height)
            bottom_right = (bottom_right_width, bottom_right_height)
        
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
        
            image = Image.fromarray(image)
            draw = ImageDraw.Draw(image)
            draw.polygon([top_left, top_right, bottom_right, bottom_left], 
                fill=(255, 255, 255), outline=(255, 255, 255))
        
            filename = os.path.join(save_dir, '{:06d}.png'.format(num_image))
            image.save(filename, 'PNG')
        
            df = df.append(
                pd.DataFrame(
                    [[filename, top_left_height, top_left_width, 
                      top_right_height, top_right_width,
                      bottom_left_height, bottom_left_width, 
                      bottom_right_height, bottom_right_width]], 
                    columns=csv_columns), ignore_index=True)
        
            print('Creating {:06d} image.'.format(num_image))
            num_image += 1
            if num_image == NUM_IMAGE:
                break
                
        df.to_csv(csv_file)
        print('Finished!')


if __name__ == '__main__':
    main()