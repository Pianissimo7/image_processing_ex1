"""
        '########:'##::::'##::::'##:::
         ##.....::. ##::'##:::'####:::
         ##::::::::. ##'##::::.. ##:::
         ######:::::. ###::::::: ##:::
         ##...:::::: ## ##:::::: ##:::
         ##:::::::: ##:. ##::::: ##:::
         ########: ##:::. ##::'######:
        ........::..:::::..:::......::
"""
from typing import List
import numpy as np
import matplotlib.pyplot as plt
import math

import cv2

LOAD_GRAY_SCALE = 1
LOAD_RGB = 2


def myID() -> np.int_:
    """
    Return my ID (not the friend's ID I copied from)
    :return: int
    """
    return 324076066


def imReadAndConvert(filename: str, representation: int) -> np.ndarray:
    path = filename

    src = cv2.imread(path)

    if representation == LOAD_GRAY_SCALE:
            image = cv2.cvtColor(src, cv2.COLOR_BGRA2GRAY)
    elif representation == LOAD_RGB:
            image = cv2.cvtColor(src, cv2.COLOR_BGRA2RGB)
    else:
        raise Exception("invalid representation code")

    normalized = cv2.normalize(image, None, 0, 1, cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    return normalized


def imDisplay(filename: str, representation: int):

    image = imReadAndConvert(filename, representation)

    if representation == LOAD_GRAY_SCALE:
        plt.imshow(image, cmap="gray")
    else:
        plt.imshow(image)


def transformRGB2YIQ(imgRGB: np.ndarray) -> np.ndarray:

    matrix = np.array([[0.299, 0.587, 0.114],
                        [0.59590059, -0.27455667, -0.32134392],
                        [0.21153661, -0.52273617, 0.31119955]])

    OrigShape = imgRGB.shape
    return np.dot(imgRGB.reshape(-1, 3), matrix.transpose()).reshape(OrigShape)


def transformYIQ2RGB(imgYIQ: np.ndarray) -> np.ndarray:
    matrix = np.array([[0.299, 0.587, 0.114],
                        [0.59590059, -0.27455667, -0.32134392],
                        [0.21153661, -0.52273617, 0.31119955]])
    OrigShape = imgYIQ.shape
    return np.dot(imgYIQ.reshape(-1, 3), np.linalg.inv(matrix).transpose()).reshape(OrigShape)


def hsitogramEqualize(imgOrig: np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray):

    width = imgOrig.shape[0]
    height = imgOrig.shape[1]

    isrgb = imgOrig.ndim == 3
    if isrgb:
        img = transformRGB2YIQ(imgOrig)[:, :, 0]
        img = np.int32(np.floor(img * 255)).flatten()
    else:
        img = np.int32(np.floor(imgOrig * 255)).flatten()

    hist = np.histogram(img, bins=256, range=(0, 256))
    cumsum = np.cumsum(hist[0])
    pixel_amount = len(img)
    LUT = [(cumsum[i] / pixel_amount * 255) for i in range(0, len(cumsum))]

    for i in range(0, len(img)):
        img[i] = LUT[img[i]]

    img = img.reshape(width, height)
    histEQ = np.histogram(img, bins=256, range=(0, 256))

    if isrgb:
        img = cv2.normalize(img, None, 0, 1, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        imgYIQ = transformRGB2YIQ(imgOrig)
        imgYIQ[:, :, 0] = img
        img = imgYIQ
        img = transformYIQ2RGB(img)

    return img, hist[0], histEQ[0]


def quantizeImage(imOrig: np.ndarray, nQuant: int, nIter: int) -> (List[np.ndarray], List[float]):

    img = imOrig.copy()

    width = imOrig.shape[0]
    height = imOrig.shape[1]

    isrgb = imOrig.ndim == 3
    if isrgb:
        img = transformRGB2YIQ(img)[:, :, 0]
        img = np.int32(np.floor(img * 255)).flatten()
    else:
        img = np.int32(np.floor(img * 255)).flatten()

    imOrig_compare = img.copy()

    imgs = []
    errors = []

    # initialize z to have boundaries of size 256 / nQuant
    z = []
    for i in range(0, nQuant):
        z.append(math.floor(i * (256 / nQuant)))
    z.append(255)

    hist = np.histogram(img, bins=256, range=(0, 256))

    plt.figure()
    plt.plot(hist[0], 'r')
    plt.show()

    for i in range(0, nIter):

        q = []

        for j in range(0, nQuant):
            start_index = z[j]
            end_index = z[j + 1]

            partition_values = hist[0][start_index:end_index]
            partition_indexes = hist[1][start_index:end_index]

            mean_index = np.average(partition_indexes, weights=partition_values)

            q.append(mean_index)

        # change every value in z to be in between 2 consecutive q values
        for j in range(len(q) - 1):
            z[j + 1] = round((q[j] + q[j + 1]) / 2)

        imgnew = imOrig_compare.copy()
        for j in range(len(z) - 1):
            imgnew[imOrig_compare >= z[j]] = q[j]

        error = np.sqrt(np.sum(np.power(imOrig_compare - imgnew, 2))) / (height * width)
        errors.append(error)

        if not isrgb:
            imgnew = imgnew.reshape(width, height)
        else:
            imgnew = cv2.normalize(imgnew, None, 0, 1, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
            imgYIQ = transformRGB2YIQ(imOrig)
            imgYIQ[:, :, 0] = imgnew.reshape((width, height))
            imgnew = imgYIQ
            imgnew = transformYIQ2RGB(imgnew)

        imgs.append(imgnew.copy())

    return imgs, errors
