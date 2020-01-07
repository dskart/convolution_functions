import convolution as cf
import numpy as np

from PIL import Image
import matplotlib.pyplot as plt


def grey_scale_convolution():
    image = np.array(Image.open('5.1.09.tiff'))
    plt.imshow(image, cmap='gray')
    plt.show()

    kernel = np.array([
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]])

    convolved_image = cf.convolve_greyscale(image, kernel)
    plt.imshow(convolved_image, cmap='gray')
    plt.show()


def rgb_convolution():
    image = np.array(Image.open('4.1.07.tiff'))
    plt.imshow(image)
    plt.show()

    kernel = np.ones((11, 11))
    kernel /= np.sum(kernel)

    convolved_image = cf.convolve_rgb(image, kernel)
    plt.imshow(convolved_image.astype('uint8'))
    plt.show()


def max_pooling_shrink():
    image = np.array(Image.open('5.1.09.tiff'))
    plt.imshow(image, cmap='gray')
    plt.show()

    kernel_size = (2, 2)
    stride = (2, 2)

    output = cf.max_pooling(image, kernel_size, stride)
    plt.imshow(output, cmap='gray')
    plt.show()


def max_pooling_blur():
    image = np.array(Image.open('5.1.09.tiff'))
    plt.imshow(image, cmap='gray')
    plt.show()

    kernel_size = (3, 3)
    stride = (1, 1)

    output = cf.max_pooling(image, kernel_size, stride)
    plt.imshow(output, cmap='gray')
    plt.show()


def max_pooling_stretch():
    image = np.array(Image.open('5.1.09.tiff'))
    plt.imshow(image, cmap='gray')
    plt.show()

    kernel_size = (3, 3)
    stride = (1, 3)

    output = cf.max_pooling(image, kernel_size, stride)
    plt.imshow(output, cmap='gray')
    plt.show()


def average_pooling_shrink():
    image = np.array(Image.open('5.1.09.tiff'))
    plt.imshow(image, cmap='gray')
    plt.show()

    kernel_size = (2, 2)
    stride = (2, 2)

    output = cf.average_pooling(image, kernel_size, stride)
    plt.imshow(output, cmap='gray')
    plt.show()


def average_pooling_blur():
    image = np.array(Image.open('5.1.09.tiff'))
    plt.imshow(image, cmap='gray')
    plt.show()

    kernel_size = (3, 3)
    stride = (1, 1)

    output = cf.average_pooling(image, kernel_size, stride)
    plt.imshow(output, cmap='gray')
    plt.show()


def average_pooling_stretch():
    image = np.array(Image.open('5.1.09.tiff'))
    plt.imshow(image, cmap='gray')
    plt.show()

    kernel_size = (3, 3)
    stride = (1, 3)

    output = cf.average_pooling(image, kernel_size, stride)
    plt.imshow(output, cmap='gray')
    plt.show()


if __name__ == "__main__":
    grey_scale_convolution()
    rgb_convolution()

    max_pooling_shrink()
    max_pooling_blur()
    max_pooling_stretch()

    average_pooling_shrink()
    average_pooling_blur()
    average_pooling_stretch()
