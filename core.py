import numpy as np
from scipy.signal import convolve2d

from utils import load_image, display_image, timeit


def rgb2gray(array):
    """
    Magic function to convert RGB image array to gray-scale

    :param array: ndarray of shape (m, n, 3)
    :return: gray-scale image array of shape (m, n)
    """
    return np.matmul(array, [.299, .587, .114])


def hog(array):
    """
    Calculate the normalized HOG (histogram of oriented gradients) descriptors of a given image. Note that the
    algorithm uses a slight variation of HOG to represent the energy of each pixel:
        the algorithm cares only about the magnitude of the gradients rather than the direction
    Explanation of HOG: https://en.wikipedia.org/wiki/Histogram_of_oriented_gradients

    :param array: input image as ndarray
    :return: the corresponding HOG which holds the same shape as the input image
    """
    horizontal_kernel = np.array([[-1, 0, 1]])  # the horizontal kernel to convolve the input image with
    vertical_kernel = horizontal_kernel.T  # the vertical convolution kernel

    # expand the gray-scale image's dimension to be consistent with that of RGB images
    if array.ndim == 2:
        array = array.reshape((*array.shape, 1))

    energy_tensor = np.zeros(array.shape)
    for ch in range(array.shape[2]):
        # calculate the energy for each channel (layer) of the image
        channel = array[:, :, ch]
        dx = convolve2d(channel, horizontal_kernel, mode='same')  # convolution to calculate horizontal gradients
        dy = convolve2d(channel, vertical_kernel, mode='same')  # calculate vertical gradients
        d = np.abs(dx) + np.abs(dy)  # variation of the HOG
        d /= d.max()  # normalization
        energy_tensor[:, :, ch] = d

    out = np.sum(energy_tensor, axis=2)
    return out


# @timeit
def find_path(array):
    """
    find the path from the top of the image to the bottom which contains minimum amount of energy amongst all paths.
    A path is also called a seam.
    The strategy to finding the path is dynamic programming:
    f(i, j) = energy(i, j) + min{f(i - 1, j - 1), f(i - 1, j), f(i - 1, j + 1)}

    :param array: the energy matrix
    :return: a path represented as python list: List[(x, y)]
    """
    dp = np.zeros(shape=array.shape)
    trace = np.zeros(shape=array.shape, dtype=np.int32)
    m, n = array.shape

    # pre-process to have the initial DP values
    for j in range(n):
        dp[0, j] = array[0, j]

    # finding the seam using dynamic programming algorithm
    for i in range(1, m):
        for j in range(n):
            dp[i, j] = dp[i - 1, j]
            trace[i, j] = 0
            if j > 0 and dp[i - 1, j - 1] < dp[i, j]:
                dp[i, j] = dp[i - 1, j - 1]
                trace[i, j] = -1
            if j + 1 < n and dp[i - 1, j + 1] < dp[i, j]:
                dp[i, j] = dp[i - 1, j + 1]
                trace[i, j] = 1
            dp[i, j] += array[i, j]

    # locate the optimal solution
    min_energy = dp[m - 1, 0]
    p = 0
    for i in range(1, n):
        if dp[m - 1, i] < min_energy:
            min_energy = dp[m - 1, i]
            p = i

    # retrieve the optimal path
    path = [(m - 1, p)]
    for i in range(m - 1, 0, -1):
        step = p + trace[i, p]
        path.append((i - 1, step))
        p += trace[i, p]

    return path


def carve(array, path):
    """
    Carve out the given seam (path) from the image
    :param array: the image array to carve the seam out from
    :param path: a seam represented as a list of tuples.
    :return: Carved out image
    """
    if array.ndim == 2:
        array = array.reshape((*array.shape, 1))
    m, n = array.shape[:2]

    output = np.zeros(shape=(m, n - 1, array.shape[2]))
    for ch in range(array.shape[2]):
        for x, y in path:
            output[x, :, ch] = np.delete(array[x, :, ch], y)
    return output.squeeze()


def reduce_width(origin, width):
    """
    Wrapper function to reduce a certain width from the given image

    :param origin: image represented as ndarray
    :param width: width to reduce
    :return: width reduced image
    """
    output = np.copy(origin)
    for _ in range(width):
        energy_matrix = hog(output)
        seam = find_path(energy_matrix)
        output = carve(output, seam)
    return output


def reduce_height(origin, height):
    """
    Wrapper function to reduce heights from the image. It's the equivalence of transposing the original image
    and apply `reduce_width` then transposing back

    :param origin: input image
    :param height: height to be reduced
    :return: reduced image
    """
    rotated = np.transpose(origin, axes=[1, 0, 2])
    output = reduce_width(rotated, height)
    return np.transpose(output, axes=[1, 0, 2])


def down_size(origin, width, height):
    """
    Wrapper function to reduce both width and height from the input image

    :param origin: input image as ndarray
    :param width: width to be reduced
    :param height: height to be reduced
    :return: reduced image
    """
    narrow = reduce_width(origin, width)
    shallow = reduce_height(narrow, height)
    return shallow
