import cv2
import matplotlib.pyplot as plt
from RegionGrowing import *

# Q3 - Region Growing
image = cv2.imread("isch_head.bmp", flags=0)
seed = (200, 100)

rg = RegionGrow(image, seed)

plt.imshow(rg.obj, cmap='gray')
plt.title(f"Object Area: {int(rg.obj_area)} [pixels]")
