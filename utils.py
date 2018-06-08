from datetime import datetime

import numpy as np
from PIL import Image
from matplotlib import pyplot as plt


def timeit(func):
    """decorator function to show time consumption of a function execution
    """
    def wrapper(*args, **kwargs):
        tick = datetime.now()
        results = func(*args, **kwargs)
        tock = datetime.now()
        print("Function: `{}` | Time elapsed: {}".format(func.__name__, tock - tick))
        return results
    return wrapper


def load_image(path):
    """
    Load from the file and output the image as an numpy array
    :param path: path to file
    :return: ndarray, the image
    """
    image = Image.open(path)
    array = np.asarray(image, dtype=np.float32)
    return array / 255


def display_image(*images, col=None):
    """
    plot an arbitrary number of images

    :param images: any number of ndarrays, each of which is an image representation
    :param col: the number of columns when displaying the images
    :return: None
    """
    if col is None:
        col = len(images)
    plt.figure(figsize=[16, 9])
    row = np.ceil(len(images) / col)
    for i, image in enumerate(images):
        plt.subplot(row, col, i + 1)
        plt.imshow(image, cmap='gray')
    plt.show()
