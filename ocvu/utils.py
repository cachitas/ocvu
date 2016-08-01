# -*- coding: utf-8 -*-

import numpy as np
import cv2


def _mean_squared_error(img1, img2):
    """
    The **Mean Squared Error** between the two images is the sum of the
    squared difference between the two images. The lower the error, the
    more *similar* the two images are.
    **NOTE:** the two images must have the same dimension.
    """
    err = np.sum((img1.astype("float") - img2.astype("float")) ** 2)
    err /= float(img1.shape[0] * img1.shape[1])
    return err


def imshow(window_name, image, border_color=(0, 0, 0)):
    """Improves default OpenCV imshow method by adding the possibility
    to correctly visualize small images without them being stretched.
    This is accomplished by adding a border around the image.
    """
    minimum_window_width = 400
    minimum_window_height = 50

    if len(image.shape) == 2:
        height, width = image.shape
    elif len(image.shape) == 3:
        height, width, _ = image.shape
    else:
        raise ValueError('Invalid image: wrong shape')

    if width < minimum_window_width:
        horizontal_border = (minimum_window_width - width) / 2
    else:
        horizontal_border = 0

    if height < minimum_window_height:
        vertical_border = (minimum_window_height - height) / 2
    else:
        vertical_border = 0

    horizontal_border = int(horizontal_border)
    vertical_border = int(vertical_border)

    if any((horizontal_border != 0, vertical_border != 0)):
        image = cv2.copyMakeBorder(image,
                                   top=vertical_border,
                                   bottom=vertical_border,
                                   left=horizontal_border,
                                   right=horizontal_border,
                                   borderType=cv2.BORDER_CONSTANT,
                                   value=border_color)
    cv2.imshow(window_name, image)
