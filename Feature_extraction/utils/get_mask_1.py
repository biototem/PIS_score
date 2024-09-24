import numpy as np
from skimage.color import rgb2hsv
from skimage.filters import threshold_otsu

def run_mask(img_RGB_array,contour_area_threshold):
    img_HSV = rgb2hsv(img_RGB_array)
    background_R = img_RGB_array[:, :, 0] > threshold_otsu(img_RGB_array[:, :, 0])
    background_G = img_RGB_array[:, :, 1] > threshold_otsu(img_RGB_array[:, :, 1])
    background_B = img_RGB_array[:, :, 2] > threshold_otsu(img_RGB_array[:, :, 2])
    tissue_RGB = np.logical_not(background_R & background_G & background_B)
    tissue_S = img_HSV[:, :, 1] > threshold_otsu(img_HSV[:, :, 1])
    min_R = img_RGB_array[:, :, 0] > 50
    min_G = img_RGB_array[:, :, 1] > 50
    min_B = img_RGB_array[:, :, 2] > 50
    tissue_mask = tissue_S & tissue_RGB & min_R & min_G & min_B
    return np.uint8(tissue_mask)


