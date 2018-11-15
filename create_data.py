from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from PIL import Image, ImageDraw
import random
import os
import pandas as pd


IMAGE_SHAPE = (224, 224, 3)
HEIGHT = IMAGE_SHAPE[0]
WIDTH = IMAGE_SHAPE[1]
NUM_IMAGES = [5000, 1500, 2000]

SAVE_DIRS = ['./mask_data/quadrilateral/train', 
             './mask_data/quadrilateral/val',
             './mask_data/quadrilateral/test']
CSV_FILES = ['./mask_data/quadrilateral/train.csv',
             './mask_data/quadrilateral/val.csv',
             './mask_data/quadrilateral/test.csv']
CSV_COLUMNS = ['filename', 'top_left_height', 'top_left_width', 
               'top_right_height', 'top_right_width',
               'bottom_left_height', 'bottom_left_width', 
               'bottom_right_height', 'bottom_right_width']


def create_rectangle_mask():
    """Compute images with one arbitrary rectangle mask.
    """
    for save_dir, csv_file, NUM_IMAGE in zip(SAVE_DIRS, CSV_FILES, NUM_IMAGES):
        num_image = 0
        df = pd.DataFrame(columns=CSV_COLUMNS)
        while True:
            image = np.zeros(IMAGE_SHAPE, np.uint8)
            # background is (128, 128, 128)
            image.fill(128)
        
            # create mask coordinates
            top_left_height = random.randint(0, HEIGHT - 1)
            top_left_width = random.randint(0, WIDTH - 1)
            top_right_height = top_left_height
            top_right_width = random.randint(top_left_width, WIDTH - 1)
            bottom_left_height = random.randint(top_left_height, HEIGHT - 1)
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
                    columns=CSV_COLUMNS), ignore_index=True)
        
            print('Creating {:06d} image.'.format(num_image))
            num_image += 1
            if num_image == NUM_IMAGE:
                break
                
        df.to_csv(csv_file)
        print('Finished!')


def create_quadrilateral_mask():
    """Create images with one convex quadrilateral mask.
    """
    for save_dir, csv_file, NUM_IMAGE in zip(SAVE_DIRS, CSV_FILES, NUM_IMAGES):
        num_image = 0
        df = pd.DataFrame(columns=CSV_COLUMNS)
        while True:
            image = np.zeros(IMAGE_SHAPE, np.uint8)
            # background is (128, 128, 128)
            image.fill(128)
        
            # create mask coordinates
            top_left_height = random.randint(5, HEIGHT - 5)
            top_left_width = random.randint(5, WIDTH - 5)
            top_right_height = top_left_height
            top_right_width = random.randint(top_left_width, WIDTH - 5)
            bottom_left_height = random.randint(top_left_height, HEIGHT - 5)
            bottom_left_width = top_left_width
            bottom_right_height = bottom_left_height
            bottom_right_width = top_right_width
        
            assert top_right_width >= top_left_width
            assert bottom_left_height >= top_left_height
        
            if bottom_right_height - top_left_height < 35 \
                or bottom_right_width - top_left_width < 35:
                continue
        
            # Point is represented as (width, height) in PIL.
            top_left = (top_left_width, top_left_height)
            top_right = (top_right_width, top_right_height)
            bottom_left = (bottom_left_width, bottom_left_height)
            bottom_right = (bottom_right_width, bottom_right_height)

            def random_point(point):
                width, height = point
                width += random.randint(-12, 12)
                height += random.randint(-12, 12)

                width = 0 if width < 0 else width
                height = 0 if height < 0 else height

                width = WIDTH - 1 if width >= WIDTH else width
                height = HEIGHT - 1 if height >= HEIGHT else height

                return (width, height)
            
            corner_points = [top_left, top_right, bottom_right, bottom_left]
            corner_points = [random_point(point) for point in corner_points]
        
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
        
            image = Image.fromarray(image)
            draw = ImageDraw.Draw(image)
            draw.polygon(corner_points, fill=(255, 255, 255), 
                outline=(255, 255, 255))
        
            filename = os.path.join(save_dir, '{:06d}.png'.format(num_image))
            image.save(filename, 'PNG')

            # corner_points = [top_left, top_right, bottom_left, bottom_right]
            corner_points = [list(point)[::-1] for point in corner_points]
            corner_points = corner_points[0:2] + corner_points[-2:][::-1]
            # corner_points = [top_left_height, top_left_width,
            #                  top_right_height, top_right_width,
            #                  bottom_left_height, bottom_left_width,
            #                  bottom_right_height, bottom_right_width]
            corner_points = [x for point in corner_points for x in point]
        
            df = df.append(
                pd.DataFrame(
                    [[filename] + corner_points], 
                    columns=CSV_COLUMNS), ignore_index=True)
        
            print('Creating {:06d} image.'.format(num_image))
            num_image += 1
            if num_image == NUM_IMAGE:
                break
                
        df.to_csv(csv_file)
        print('Finished!')


def main():
    create_quadrilateral_mask()


if __name__ == '__main__':
    main()
