from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random

import numpy as np


class Quadrilateral(object):
    """Class for manipulating quadrilateral.
    """
    def __init__(self, quadrilateral=None, image_shape=(224, 224, 3)):
        """Constructor.

        Args:
            quadrilateral: an integer list representing a quadrilateral
                or an emtpy list.
            image_shape: the shape of image. Default to (224, 224, 3).
        """
        self._image_shape = image_shape
        self._height = image_shape[0]
        self._width = image_shape[1]
        self._channel = image_shape[2]
        if quadrilateral is not None:
            self._quadrilateral = quadrilateral
        else:
            self._quadrilateral = self.create_quadrilateral()

    def get(self):
        """Convenience function for accessing corner points of quadrilateral.

        Returns:
            a list of corner points:
            [top_left, top_right, bottom_left, bottom_right]. Each point is
            represented as (height, width).
        """
        return self._quadrilateral

    def get_pil_format(self):
        """Get coordinates of corner points in pillow format.

        Point in pillow is represented as (width, height). This function
        changes the order of last two points and exchange the order of 
        height and width of each point.

        Returns:
            corner_points: a list of corner points in pillow format:
            [top_left, top_right, bottom_right, bottom_left].
        """
        corner_points = self._quadrilateral
        corner_points = corner_points[:2] + corner_points[2:][::-1]
        corner_points = [point[::-1] for point in corner_points]

        return corner_points
    
    def get_csv_format(self):
        """Get coordinates of corner points which can be written to 
        csv file directly.

        Returns:
            corner_points:
        """
        corner_points = self._quadrilateral
        corner_points = [x for point in corner_points for x in point]

        return corner_points

    def decode_csv_format(self):
        """Decode coordinates from csv formated coordinates.
        """
        # TODO(hhw): write code to decode csv formated coordinates.
        pass
    
    def update(self):
        """Update the attribute `_quadrilateral` in place.
        """
        self._quadrilateral = self.create_quadrilateral()

    def reset(self):
        """Reset attribute `_quadrilateral`.
        """
        self._quadrilateral = []

    def create_quadrilateral(self):
        """Create one quadrilateral.

        Returns:
            corner_points: list of corner point coordinates.
                [top_left, top_right, bottom_left, bottom_right].
                Each corner point is represented as (heihgt, width).
        """
        corner_points = self.create_rectangle()
        corner_points = [self._random_shift(point) for point in corner_points]

        return corner_points

    def create_rectangle(self):
        """Create one rectangle.

        Returns:
            corner_points: list of corner point coordinates.
                [top_left, top_right, bottom_left, bottom_right].
                Each corner point is represented as (heihgt, width).
        """
        while True:
            top_left_height = random.randint(5, self._height - 5)
            top_left_width = random.randint(5, self._width - 5)
            top_right_height = top_left_height
            top_right_width = random.randint(top_left_width, self._width - 5)
            bottom_left_height = random.randint(top_left_height, self._height - 5)
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

        top_left = (top_left_height, top_left_width)
        top_right = (top_right_height, top_right_width)
        bottom_left = (bottom_left_height, bottom_left_width)
        bottom_right = (bottom_right_height, bottom_right_width)

        corner_points = [top_left, top_right, bottom_left, bottom_right]

        return corner_points

    def is_overlap(self, quadrilateral):
        """Check whether two quadrilaterals overlap with each other.

        Args:
            quadrilateral: an instance of Quadrilateral.

        Returns:
            is_overlap: Boolean indicates whether two quadrilateral
                are overlapped.

        Raises:
            TypeError: Type of argument quadrilateral is not Quadrilateral.
            ValueError: self._quadrilateral or quadrilateral._quadrilateral
                        is emtpy list.
        """
        if not isinstance(quadrilateral, type(self)):
            raise TypeError(
                'Argument quadrilateral should be instance of {}, which actually '
                'has type {}'.format(type(self), type(quadrilateral)))
        if not len(self._quadrilateral) or not len(quadrilateral._quadrilateral):
            raise ValueError('The `_quadrilateral` should not be empty.')

        mask1 = self.get_mask()
        mask2 = quadrilateral.get_mask()

        mask = mask1 * mask2
        mask = mask.astype(np.bool)
        is_overlap = np.sum(mask)

        return bool(is_overlap)

    def is_soft_overlap(self, quadrilateral):
        """Check whether two quadrilaterals overlap softly with each other.

        Args:
            quadrilateral: an instance of Quadrilateral.

        Returns:
            is_overlap: Boolean indicates whether two quadrilateral
                are overlapped.

        Raises:
            TypeError: Type of argument quadrilateral is not Quadrilateral.
            ValueError: self._quadrilateral or quadrilateral._quadrilateral
                        is emtpy list.
        """
        if not isinstance(quadrilateral, type(self)):
            raise TypeError(
                'Argument quadrilateral should be instance of {}, which actually '
                'has type {}'.format(type(self), type(quadrilateral)))
        if not len(self._quadrilateral) or not len(quadrilateral._quadrilateral):
            raise ValueError('The `_quadrilateral` should not be empty.')

        mask1 = self.get_mask()
        mask2 = quadrilateral.get_mask()

        mask = mask1 * mask2
        mask = mask.astype(np.bool)
        threshold = 1000
        num_overlap_pixel = np.sum(mask)

        if num_overlap_pixel > threshold:
            is_overlap = True
        else:
            is_overlap = False

        return is_overlap

    def get_mask(self):
        """Get boolean mask indicates quadrilateral area.

        Returns:
            mask: Numpy ndarray. A boolean mask indicates quadrilateral area.
        """
        mask = np.zeros((self._height, self._width), dtype=np.bool)

        y_min, x_min, y_max, x_max = self.get_bounding_box()

        for h in range(y_min, y_max + 1):
            for w in range(x_min, x_max + 1):
                if self.is_point_inside((w, h)):
                    mask[h][w] = True

        return mask

    def get_bounding_box(self):
        """Compute bounding box of quadrilateral.

        Returns:
            bounding_box: List of bounding box corner points.

        Raises:
            ValueError: self._quadrilateral is an emtpy list.
        """
        if not len(self._quadrilateral):
            raise ValueError('The `_quadrilateral` should not be empty list.')

        # each point is represented as (height, width)
        top_left, top_right, bottom_left, bottom_right = self.get()
        y_min = min(top_left[0], top_right[0])
        y_max = max(bottom_left[0], bottom_right[0])
        x_min = min(top_left[1], bottom_left[1])
        x_max = max(top_right[1], bottom_right[1])

        return [y_min, x_min, y_max, x_max]

    def is_point_inside(self, point):
        """Check whether a point is inside the quadrilateral.

        Args:
            point: a point represented as (height, width).

        Returns:
            is_inside: Boolean indicates whether the point is inside
                the quadrilateral.

        Raises:
            ValueError: if length of argument point is not 2.
            ValueError: self._quadrilateral is an emtpy list.
        """
        if len(point) != 2:
            raise ValueError('Argument point is expected of length 2.')
        if not len(self._quadrilateral):
            raise ValueError('The `_quadrilateral` should not be empty list.')

        point = np.array(point)
        corner_points = np.array(self._quadrilateral)
        # A - top_left, B - top_right, C - bottom_right, D - bottom_left
        A, B, D, C = map(np.squeeze, np.split(corner_points, 4, axis=0))

        a = self._compute_cross_product(point - A, B - A)
        b = self._compute_cross_product(point - B, C - B)
        c = self._compute_cross_product(point - C, D - C)
        d = self._compute_cross_product(point - D, A - D)

        if ((a >= 0 and b >= 0 and c >= 0 and d >= 0)
            or (a <= 0 and b <= 0 and c <= 0 and d <= 0)):
            is_inside = True
        else:
            is_inside = False

        return is_inside

    def _random_shift(self, point):
        """Shift point randomly.

        Args:
            point: tuple of coordinates which is (height, width).

        Returns:
            new_point: shifted coordinates of the point.
        """
        height, width = point
        height += random.randint(-14, 14)
        width += random.randint(-14, 14)

        height = 0 if height < 0 else height
        width = 0 if width < 0 else width
        height = self._height - 1 if height >= self._height else height
        width = self._width - 1 if width >= self._width else width
        new_point = (height, width)

        return new_point

    def _compute_cross_product(self, vector1, vector2):
        """Compute cross product of point vector1 and vector2.

        Args:
            vector1: A point represented as [height, width].
            vector2: A point represented as [height, width].

        Returns:
            cross_product: Cross product of vector1 and vector2.
        """
        # compute cross product: x1y2 - x2y1
        cross_product = vector1[1] * vector2[0] - vector1[0] * vector2[1]

        return cross_product

    def __len__(self):
        """Used to convert to boolean.
        """
        return len(self._quadrilateral)

    def __repr__(self):
        return 'Instance of Quadrilateral:' + str(self._quadrilateral)

    __str__ = __repr__
