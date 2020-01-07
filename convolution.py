import numpy as np
import math


class Convolution():
    def __init__(self, kernel):
        self._kernel = kernel

        self._kernel_y_size, self._kernel_x_size = kernel.shape
        self._x_padding = self._kernel_x_size // 2
        self._y_padding = self._kernel_y_size // 2
        self.flipped_kernel = self.flip_kernel()

    def flip_kernel(self):
        new_kernel = np.flip(self._kernel, 0)
        new_kernel = np.flip(new_kernel, 1)
        return new_kernel

    def pad_image(self, image):
        padded_image = np.pad(image, ((self._y_padding, self._y_padding), (self._x_padding, self._x_padding)),
                              'constant', constant_values=(0))

        return padded_image

    def covolve_pixel(self, x_pos, y_pos, image, kernel):

        x_start = x_pos-self._x_padding
        y_start = y_pos-self._y_padding

        y_end = y_pos + self._y_padding + 1
        x_end = x_pos + self._x_padding + 1

        image_kernel_patch = kernel * image[y_start:y_end, x_start:x_end]
        return np.sum(image_kernel_patch)

    def convolve_image_greyscale(self, image):

        padded_image = self.pad_image(image)

        y_image_end, x_image_end = image.shape

        convolved_image = np.zeros([y_image_end, x_image_end])

        for y in range(self._y_padding, y_image_end + self._y_padding):
            for x in range(self._x_padding, x_image_end + self._x_padding):

                convolved_image[y-self._y_padding,
                                x-self._x_padding] = self.covolve_pixel(
                    x, y, padded_image, self.flipped_kernel)

        return convolved_image

    def convolve_image_rgb(self, image):
        Y, X, depth = image.shape

        convolved_depths = np.zeros([Y, X, depth])
        for d in range(0, depth):
            convolved_depths[:, :, d] = self.convolve_image_greyscale(
                image[:, :, d])

        return convolved_depths


def convolve_greyscale(image, kernel):
    convolution = Convolution(kernel)

    return convolution.convolve_image_greyscale(image)


def convolve_rgb(image, kernel):
    convolution = Convolution(kernel)

    return convolution.convolve_image_rgb(image)


def max_pooling(image, kernel, stride):
    image_height, image_width = image.shape

    kernel_height, kernel_width = kernel
    stride_height, stride_width = stride
    y = 0
    x = 0

    max_pooling_array = []
    while (y+kernel_height) <= image_height:
        max_pooling_x = []
        while (x+kernel_width) <= image_width:
            max_value = np.max(image[y:(y+kernel_height), x:(x+kernel_width)])
            max_pooling_x.append(max_value)
            x += stride_width

        max_pooling_array.append(max_pooling_x)
        y += stride_height
        x = 0

    return np.array(max_pooling_array)


def average_pooling(image, kernel, stride):
    image_height, image_width = image.shape

    kernel_height, kernel_width = kernel
    stride_height, stride_width = stride
    y = 0
    x = 0

    max_pooling_array = []
    while (y+kernel_height) <= image_height:
        max_pooling_x = []
        while (x+kernel_width) <= image_width:
            max_value = np.average(
                image[y:(y+kernel_height), x:(x+kernel_width)])
            max_pooling_x.append(max_value)
            x += stride_width

        max_pooling_array.append(max_pooling_x)
        y += stride_height
        x = 0

    return np.array(max_pooling_array)
