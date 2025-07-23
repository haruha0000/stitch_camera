import stitching as st
import camera as cr
import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from skimage import data
from skimage.filters import gaussian
from skimage.segmentation import active_contour


def changed_region(frame1, frame2):

    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    _, mask2 = cv2.threshold(gray2, 100, 255, cv2.THRESH_BINARY)
    _, mask1 = cv2.threshold(gray1, 30, 255, cv2.THRESH_BINARY)
    change_region = cv2.bitwise_xor(mask1, mask2)
    return change_region

def snake_detect():
    s = np.linspace(0, 2 * np.pi, 400)
    r = 100 + 100 * np.sin(s)
    c = 220 + 100 * np.cos(s)
    init = np.array([r, c]).T

    snake = active_contour(
        gaussian(img, sigma=3, preserve_range=False),
        init,
        alpha=0.015,
        beta=10,
        gamma=0.001,
    )

if __name__ == "__main__":
    img1 = cv2.imread('DSC02930.JPG')
    img2 = cv2.imread('DSC02931.JPG')
    c = changed_region(img1, img2)
    plt.imshow(c)