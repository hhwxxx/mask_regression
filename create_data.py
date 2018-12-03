from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random
import os

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFilter
from skimage import morphology


IMAGE_SHAPE = (224, 224, 3)
HEIGHT = IMAGE_SHAPE[0]
WIDTH = IMAGE_SHAPE[1]
NUM_IMAGES = [5000, 1500, 2000]

SAVE_DIRS = ['./mask_data/quadrilateral_1_2/train', 
             './mask_data/quadrilateral_1_2/val',
             './mask_data/quadrilateral_1_2/test']
CSV_FILES = ['./mask_data/quadrilateral_1_2/train.csv',
             './mask_data/quadrilateral_1_2/val.csv',
             './mask_data/quadrilateral_1_2/test.csv']

CSV_COLUMNS = ['filename', 'top_left_height', 'top_left_width', 
               'top_right_height', 'top_right_width',
               'bottom_left_height', 'bottom_left_width', 
               'bottom_right_height', 'bottom_right_width']


def create_rectangle_mask():
    """Create images with one arbitrary rectangle mask.
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
        
            if (bottom_right_height - top_left_height < 30
                    or bottom_right_width - top_left_width < 30):
                continue
        
            # Point is represented as (width, height) in PIL.
            top_left = (top_left_width, top_left_height)
            top_right = (top_right_width, top_right_height)
            bottom_left = (bottom_left_width, bottom_left_height)
            bottom_right = (bottom_right_width, bottom_right_height)

            corner_points = [top_left, top_right, bottom_right, bottom_left]

            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
        
            image = Image.fromarray(image)
            draw = ImageDraw.Draw(image)
            draw.polygon(corner_points, fill=(255, 255, 255), 
                outline=(255, 255, 255))
        
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
        
            if (bottom_right_height - top_left_height < 35
                    or bottom_right_width - top_left_width < 35):
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


def create_quadrilateral_mask_noise():
    """Create images with one convex quadrilateral mask (noise).
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
        
            if (bottom_right_height - top_left_height < 35
                    or bottom_right_width - top_left_width < 35):
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
            
            resize_height = random.randint(50, 150)
            resize_width = random.randint(50, 150)
            image = image.resize(
                (resize_width, resize_height), Image.NEAREST).resize(
                    IMAGE_SHAPE[:2], Image.NEAREST)

            # image = image.filter(ImageFilter.BLUR)
            r, g, b = image.split()
            r = morphology.opening(np.array(r), morphology.disk(5))
            image = np.stack([r, r, r], axis=2)
            image = Image.fromarray(image)
        
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


def create_quadrilateral_multiple_mask_noise():
    """Create arbitrary number of quadrilateral in one image. 

    Returns:
        corner_points: List of corner points of qudrilateral.
        Each corner point is represented as (width, height).
    """
    def _create_quadrilateral():
        """Create one quadrilateral and return its coordinates.
        
        Returns:
            corner_points: list of corner point coordinates.
            Each corner point is represented as (width, height).
        """
        def _random_point(point):
            """Shift point randomly.

            Args:
                point: tuple of coordinates which is (width, height).

            Returns:
                new_point: shifted coordinates of corner point.
            """
            width, height = point
            width += random.randint(-14, 14)
            height += random.randint(-14, 14)

            width = 0 if width < 0 else width
            height = 0 if height < 0 else height
            width = WIDTH - 1 if width >= WIDTH else width
            height = HEIGHT - 1 if height >= HEIGHT else height

            new_point = (width, height)

            return new_point

        while True:
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
        
            if (bottom_right_height - top_left_height < 35 
                    or bottom_right_width - top_left_width < 35):
                continue
            else:
                break
        
        # Point is represented as (width, height) in PIL.
        top_left = (top_left_width, top_left_height)
        top_right = (top_right_width, top_right_height)
        bottom_left = (bottom_left_width, bottom_left_height)
        bottom_right = (bottom_right_width, bottom_right_height)
        
        corner_points = [top_left, top_right, bottom_right, bottom_left]
        corner_points = [_random_point(point) for point in corner_points]

        return corner_points

    def _mask(corner_points):
        """Get boolean mask indicates quadrilateral area.

        Args:
            corner_points: List of corner points of qudrilateral.
            Each corner point is represented as (width, height).

        Returns:
            mask: Numpy ndarray. A boolean mask indicates
            quadrilateral area.
        """
        def _is_inside(point, corner_points):
            """Check whether a point is inside the quadrilateral
            determined by corner_points.

            Args:
                point: a point represented as (width, height).
                corner_points: List of corner points of qudrilateral.
                Each corner point is represented as (width, height).

            Returns:
                is_inside: Boolean indicates whether the point is inside
                the quadrilateral.
            """
            def _cross_product(P, Q):
                """Compute cross product of point P and Q.
                
                Args:
                    P: A point represented as [width, height].
                    Q: A point represented as [width, height].
                
                Returns:
                    cross_product: Cross product of point P and Q.
                """
                # compute cross product: x1y2 - x2y1
                cross_product = P[0] * Q[1] - P[1] * Q[0]

                return cross_product
            
            point = np.array(point)
            corner_points = np.array(corner_points)
            # A - top_left, B - top_right, C - bottom_right, D - bottom_left
            A, B, C, D = map(np.squeeze, np.split(corner_points, 4, axis=0))

            a = _cross_product(point - A, B - A)
            b = _cross_product(point - B, C - B)
            c = _cross_product(point - C, D - C)
            d = _cross_product(point - D, A - D)

            if ((a >= 0 and b >= 0 and c >= 0 and d >= 0)
                or (a <= 0 and b <= 0 and c <= 0 and d <= 0)):
                is_inside = True
            else:
                is_inside = False
            
            return is_inside
        
        def _bounding_box(corner_points):
            """Compute bounding box of arbitrary quadrilateral.
            
            Args:
                corner_points: List of corner points of qudrilateral.
                Each corner point is represented as (width, height).
            
            Returns:
                bounding_box: List of bounding box corner points.
            """
            # each point is represented as (width, height)
            top_left, top_right, bottom_right, bottom_left = corner_points
            y_min = min(top_left[1], top_right[1])
            y_max = max(bottom_left[1], bottom_right[1])
            x_min = min(top_left[0], bottom_left[0])
            x_max = max(top_right[0], bottom_right[0])

            bounding_box = [y_min, x_min, y_max, x_max]

            return bounding_box

        H, W = IMAGE_SHAPE[:2]
        mask = np.zeros((H, W), dtype=np.bool)

        y_min, x_min, y_max, x_max = _bounding_box(corner_points)
        
        for h in range(y_min, y_max + 1):
            for w in range(x_min, x_max + 1):
                if _is_inside((w, h), corner_points):
                    mask[h][w] = True
        
        return mask

    def _is_overlap(quadrilateral_1, quadrilateral_2):
        """Check whether two quadrilaterals overlap with each other.

        Args:
            quadrilateral_1: First quadrilateral. List of corner point 
            coordinates, which is represented as (width, height).
            quadrilateral_2: Second quadrilateral.

        Returns:
            is_overlap: Boolean indicates whether two quadrilateral
            are overlapped.
        """
        mask1 = _mask(quadrilateral_1)
        mask2 = _mask(quadrilateral_2)

        mask = mask1 * mask2
        mask = mask.astype(np.bool)
        is_overlap = np.sum(mask)

        return is_overlap

    for save_dir, csv_file, NUM_IMAGE in zip(SAVE_DIRS, CSV_FILES, NUM_IMAGES):
        num_image = 0
        df = pd.DataFrame(columns=CSV_COLUMNS)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        while True:
            image = np.zeros(IMAGE_SHAPE, np.uint8)
            # background is (128, 128, 128)
            image.fill(128)
            image = Image.fromarray(image)
            draw = ImageDraw.Draw(image)
            
            # Create 2 masks per image.
            # while True:
            #     quadrilateral_1 = _create_quadrilateral()
            #     cur_iter_num = 0
            #     max_iter_num = 5
            #     flag = False
            #     while cur_iter_num < max_iter_num:
            #         quadrilateral_2 = _create_quadrilateral()
            #         flag = _is_overlap(quadrilateral_1, quadrilateral_2)
            #         cur_iter_num += 1
            #         if flag:
            #             continue
            #         else:
            #             break
            #     if flag:
            #         continue
            #     else:
            #         break
                

            # Create 1 or 2 masks per image.
            # first quadrilateral
            quadrilateral_1 = _create_quadrilateral()
            # second quadrilateral
            threshold = 0.1
            if random.random() > threshold:
                max_iter_num = 5
                cur_iter_num = 0
                while cur_iter_num < max_iter_num:
                    quadrilateral_2 = _create_quadrilateral()
                    if _is_overlap(quadrilateral_1, quadrilateral_2):
                        cur_iter_num += 1
                    else:
                        break
                # the first quadrilateral may be very big and make it
                # hard to create second quadrilateral.
                # so let the second quadrilateral to be void
                if cur_iter_num >= max_iter_num:
                    quadrilateral_2 = []
            else:
                quadrilateral_2 = []
            

            # list of quadrilaterals
            quadrilateral_list = [quadrilateral_1, quadrilateral_2]

            for quadrilateral in quadrilateral_list:
                if quadrilateral:
                    draw.polygon(quadrilateral, fill=(255, 255, 255),
                                 outline=(255, 255, 255))
            
            # add noise
            resize_height = random.randint(50, 120)
            resize_width = random.randint(50, 120)
            image = image.resize(
                (resize_width, resize_height), Image.NEAREST).resize(
                    IMAGE_SHAPE[:2], Image.NEAREST)
            # image = image.filter(ImageFilter.BLUR)
            r, g, b = image.split()
            r = morphology.opening(np.array(r), morphology.disk(5))
            image = np.stack([r, r, r], axis=2)
            image = Image.fromarray(image)
        
            filename = os.path.join(save_dir, '{:06d}.png'.format(num_image))
            image.save(filename, 'PNG')
            
            corner_points_list = []
            for quadrilateral in quadrilateral_list:
                if quadrilateral:
                    # corner_points should be 
                    # [top_left, top_right, bottom_left, bottom_right]
                    corner_points = [list(point)[::-1] for point in quadrilateral]
                    corner_points = corner_points[0:2] + corner_points[-2:][::-1]
                    # corner_points should be
                    # [top_left_height, top_left_width, 
                    #  top_right_height, top_right_width,
                    #  bottom_left_height, bottom_left_width, 
                    #  bottom_right_height, bottom_right_width]
                    corner_points = [x for point in corner_points for x in point]
                    corner_points_list.append(corner_points)
            
            corner_points = [list(point) for point in zip(*corner_points_list)]

            df = df.append(
                pd.DataFrame([[filename] + corner_points], 
                             columns=CSV_COLUMNS),
                ignore_index=True)
        
            print('Creating {:06d} image.'.format(num_image))
            num_image += 1
            if num_image == NUM_IMAGE:
                break
                
        df.to_csv(csv_file)
        print('Finished!')


def main():
    create_quadrilateral_multiple_mask_noise()
    
    
if __name__ == '__main__':
    main()
