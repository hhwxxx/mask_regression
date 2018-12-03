from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random
import os
import shutil
import functools

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFilter
from skimage import morphology

from quadrilateral import Quadrilateral


SAVE_DIR = '../mask_data/test/quadrilateral_1_2'
MASK_TYPE = 'two_quadrilaterals'
IMAGE_SHAPE = (224, 224, 3)
HEIGHT, WIDTH, CHANNEL = IMAGE_SHAPE
NUM_IMAGES = {
    'train': 5000,
    'val': 1500,
    'test': 2000,
}
CSV_COLUMNS = [
    'filename',
    'top_left_height', 'top_left_width',
    'top_right_height', 'top_right_width',
    'bottom_left_height', 'bottom_left_width',
    'bottom_right_height', 'bottom_right_width',
]


def create_rectangle_1():
    """Create 1 rectangle per image.

    Returns:
        quadrilateral_list: List of Quadrilateral instances.
    """
    quadrilateral_1 = Quadrilateral([])
    quadrilateral_1.quadrilateral = quadrilateral_1.create_rectangle()
    quadrilateral_list = [quadrilateral_1]

    return quadrilateral_list


def create_quadrilateral_1():
    """Create 1 quadrilateral per image.

    Returns:
        quadrilateral_list: List of Quadrilateral instances.
    """
    quadrilateral_1 = Quadrilateral()
    quadrilateral_list = [quadrilateral_1]

    return quadrilateral_list


def create_quadrilateral_2(connected=False):
    """Create 2 quadrilaterals per image.

    Args:
        connected: contain connected masks or not.

    Returns:
        quadrilateral_list: List of Quadrilateral instances.
    """
    # first quadrilateral
    quadrilateral_1 = Quadrilateral([])
    # second quadrilateral
    quadrilateral_2 = Quadrilateral([])
    while True:
        quadrilateral_1.update()
        cur_iter_num = 0
        max_iter_num = 5
        flag = False
        while cur_iter_num < max_iter_num:
            quadrilateral_2.update()
            if connected:
                flag = quadrilateral_1.is_soft_overlap(quadrilateral_2)
            else:
                flag = quadrilateral_1.is_overlap(quadrilateral_2)
            cur_iter_num += 1
            if flag:
                continue
            else:
                break
        if flag:
            continue
        else:
            break
    
    quadrilateral_list = [quadrilateral_1, quadrilateral_2]

    return quadrilateral_list


def create_quadrilateral_3(connected=False):
    """Create 3 quadrilaterals per image.

    Args:
        connected: contain connected masks or not.

    Returns:
        quadrilateral_list: List of Quadrilateral instances.
    """
    quadrilateral_1 = Quadrilateral([])
    quadrilateral_2 = Quadrilateral([])
    quadrilateral_3 = Quadrilateral([])
    while True:
        quadrilateral_1.update()
        cur_iter_num = 0
        max_iter_num = 5
        flag = False
        while cur_iter_num < max_iter_num:
            quadrilateral_2.update()
            if connected:
                flag = quadrilateral_1.is_soft_overlap(quadrilateral_2)
            else:
                flag = quadrilateral_1.is_overlap(quadrilateral_2)
            cur_iter_num += 1
            if flag:
                continue
            else:
                inner_cur_iter_num = 0
                inner_max_iter_num = 8
                inner_flag = False
                while inner_cur_iter_num < inner_max_iter_num:
                    quadrilateral_3.update()
                    if connected:
                        inner_flag = (quadrilateral_3.is_soft_overlap(quadrilateral_1)
                                      or quadrilateral_3.is_soft_overlap(quadrilateral_2))
                    else:
                        inner_flag = (quadrilateral_3.is_overlap(quadrilateral_1)
                                      or quadrilateral_3.is_overlap(quadrilateral_2))
                    inner_cur_iter_num += 1
                    if inner_flag:
                        continue
                    else:
                        break
                if inner_flag:
                    continue
                else:
                    break
        if inner_flag or flag:
            continue
        else:
            break

    quadrilateral_list = [quadrilateral_1, quadrilateral_2, quadrilateral_3]

    return quadrilateral_list


def create_quadrilateral_1_2(connected=False):
    """Create 1 or 2 quadrilaterals per image.

    Args:
        connected: contain connected masks or not.

    Returns:
        quadrilateral_list: List of Quadrilateral instances.
    """
    threshold = 0.5
    if random.random() > threshold:
        # two quadrilaterals
        quadrilateral_list = create_quadrilateral_2(connected)
    else:
        # one quadrilateral
        quadrilateral_list = [Quadrilateral(), Quadrilateral([])]

    return quadrilateral_list


def create_quadrilateral_1_2_3(connected=False):
    """Create 1, 2 or 3 quadrilaterals per image.

    Args:
        connected: contain connected masks or not.

    Returns:
        quadrilateral_list: List of Quadrilateral instances.
    """
    threshold_1 = 0.33
    threshold_2 = 0.66
    random_number = random.random()
    if random_number < threshold_1:
        # three quadrilaterals
        quadrilateral_list = create_quadrilateral_3(connected)
    elif random_number > threshold_2:
        # two quadrilaterals
        quadrilateral_list = create_quadrilateral_2(connected)
        quadrilateral_list.append(Quadrilateral([]))
    else:
        # one quadrilateral
        quadrilateral_list = [Quadrilateral(), Quadrilateral([]), 
                              Quadrilateral([])]

    return quadrilateral_list


def add_noise(image):
    """Add noise to image.

    Args:
        image: instance of Image.

    Returns:
        image: instance of Image.
    """
    # resize image to add noise.
    resize_height = random.randint(50, 120)
    resize_width = random.randint(50, 120)
    image = image.resize(
        (resize_width, resize_height), Image.NEAREST).resize(
            IMAGE_SHAPE[:2], Image.NEAREST)

    # morphology operation: opening
    # image = image.filter(ImageFilter.BLUR)
    r, g, b = image.split()
    r = morphology.opening(np.array(r), morphology.disk(5))
    image = np.stack([r, r, r], axis=2)
    image = Image.fromarray(image)

    return image


def draw_image(image, quadrilateral_list):
    """Draw quadrilaterals on image

    Args:
        image: instance of Image.
        quadrilateral_list: list of Quadrilateral instances.
    """
    draw = ImageDraw.Draw(image)

    for quadrilateral in quadrilateral_list:
        if quadrilateral:
            draw.polygon(quadrilateral.get_pil_format(),
                         fill=(255, 255, 255),
                         outline=(255, 255, 255))

    return image


def get_csv_coordinates(quadrilateral_list):
    """Get csv formated coordinates.

    Args:
        quadrilateral_list: list of Quadrilateral instances.

    Returns:
        corner_points: list of csv formated coordinates.
    """
    corner_points_list = []
    for quadrilateral in quadrilateral_list:
        if quadrilateral:
            # quadrilateral should be
            # [top_left_height, top_left_width, 
            #  top_right_height, top_right_width,
            #  bottom_left_height, bottom_left_width, 
            #  bottom_right_height, bottom_right_width]
            corner_points_list.append(quadrilateral.get_csv_format())

    corner_points = [list(point) for point in zip(*corner_points_list)]

    return corner_points


def main(save_dir, csv_file, num_images, core_func):
    cur_image = 0
    df = pd.DataFrame(columns=CSV_COLUMNS)

    while cur_image < num_images:
        image = np.zeros(IMAGE_SHAPE, np.uint8)
        image.fill(128)  # background is (128, 128, 128)
        image = Image.fromarray(image)

        # list of Quadrilateral instances
        quadrilateral_list = core_func()

        # draw quadrilateral on image
        image = draw_image(image, quadrilateral_list)

        # add noise
        image = add_noise(image)

        # save image
        filename = os.path.join(save_dir, '{:06d}.png'.format(cur_image))
        image.save(filename, 'PNG')

        # get csv formated coordinates
        corner_points = get_csv_coordinates(quadrilateral_list)
        df = df.append(
            pd.DataFrame([[filename] + corner_points], 
                         columns=CSV_COLUMNS),
            ignore_index=True)

        print('Creating {:06d} image.'.format(cur_image))
        cur_image += 1

    df.to_csv(csv_file)


if __name__ == '__main__':
    if os.path.exists(SAVE_DIR):
        shutil.rmtree(SAVE_DIR)
    os.makedirs(SAVE_DIR)
    
    MASK_TYPE_MAP = {
        'one_rectangle': create_rectangle_1,
        'one_quadrilateral': create_quadrilateral_1,
        'two_quadrilaterals': create_quadrilateral_2,
        'three_quadrilaterals': create_quadrilateral_3,
        'one_or_two_quadrilaterals': create_quadrilateral_1_2,
        'one_or_two_or_three_quadrilaterals': create_quadrilateral_1_2_3,
        'two_connected_quadrilaterals': functools.partial(create_quadrilateral_2, connected=True),
        'three_connected_quadrilaterals': functools.partial(create_quadrilateral_3, connected=True),
        'one_or_two_connected_quadrilaterals': functools.partial(create_quadrilateral_1_2, connected=True),
        'one_or_two_or_three_connected_quadrilaterals': functools.partial(create_quadrilateral_1_2_3, connected=True),
    }
    if MASK_TYPE not in MASK_TYPE_MAP:
        raise ValueError('mask type must be one of supported type.')

    core_func = MASK_TYPE_MAP[MASK_TYPE]
    
    dataset = ['train', 'val', 'test']
    for dataset_split in dataset:
        save_dir = os.path.join(SAVE_DIR, dataset_split)
        os.makedirs(save_dir)
        csv_file = os.path.join(SAVE_DIR, dataset_split + '.csv')
        num_images = NUM_IMAGES[dataset_split]
        main(save_dir, csv_file, num_images, core_func)
        print('*' * 50)
        print('Finished creating', dataset_split)
        print('*' * 50)
