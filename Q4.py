import numpy as np
import matplotlib.pyplot as plt
import cv2
import scipy as sp
import scipy.ndimage


def flood_fill(image, h_max=255):
    input_img = np.copy(image)
    el = sp.ndimage.generate_binary_structure(2, 2).astype(np.int)
    in_mask = sp.ndimage.binary_erosion(~np.isnan(input_img), structure=el)
    res = np.copy(input_img)
    res[in_mask] = h_max
    out_old = np.copy(input_img)
    out_old.fill(0)
    el = sp.ndimage.generate_binary_structure(2, 1).astype(np.int)
    while not np.array_equal(out_old, res):
        out_old = np.copy(res)
        res = np.maximum(input_img, sp.ndimage.grey_erosion(res, size=(3, 3), footprint=el))
    return res


# Q4 - Image Cleaning
img = cv2.imread("pic1dirty.jpg", flags=0)
bin_img = np.zeros(img.shape, dtype="uint8")
bin_img[img > 127] = 255
clean = cv2.medianBlur(bin_img, 5)
edges = cv2.Canny(clean, 100, 200)
full = flood_fill(edges)
comps = cv2.connectedComponents(full)[1]
plt.imshow(comps, cmap='gray')


