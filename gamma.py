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
from ex1_utils import *
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import numpy as np


def gammaDisplay(img_path: str, rep: int):
    gamma = 1
    img_subplot, slider_subplot = plt.subplots()
    img_subplot.subplots_adjust(left=0.25)
    axamp = img_subplot.add_axes([0.1, 0.25, 0.0225, 0.63])
    gamma_slider = Slider(
        ax=axamp,
        label="Gamma",
        valmin=0.1,
        valmax=2,
        valinit=gamma,
        orientation="vertical"
    )

    isrgb = rep != LOAD_GRAY_SCALE

    if not isrgb:
        img = imReadAndConvert(img_path, LOAD_GRAY_SCALE)
        img = np.power(img, 1 / gamma)
    else:
        img = imReadAndConvert(img_path, LOAD_RGB)
        img = transformRGB2YIQ(img)
        img[:, :, 0] = np.power(img[:, :, 0], 1 / gamma)
        img = transformYIQ2RGB(img)

    plt.subplot(111)
    plt.imshow(img, cmap="gray")

    if isrgb:
        imDisplay(img_path, LOAD_RGB)
    else:
        imDisplay(img_path, LOAD_GRAY_SCALE)

    def update_gamma(value):
        nonlocal gamma
        gamma = value
        nonlocal img
        gamma_corrected_img = np.power(img, 1/gamma)
        plt.imshow(gamma_corrected_img, cmap="gray")

    gamma_slider.on_changed(update_gamma)
    plt.show()


def main():
    gammaDisplay('big-cute-eyes-cat.png', LOAD_RGB)


if __name__ == '__main__':
    main()
