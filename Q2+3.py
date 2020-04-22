import numpy as np
import cv2
import matplotlib.pyplot as plt


# Q2:
# im1 - T_initial is set to be: mean(img)
def iterative_threshold(img):
    t_initial = int(round(np.mean(img)))
    t_new, t1 = t_initial, 0
    while t1 != t_new:
        t1 = t_new
        mean_low = np.mean(img[img < t1])
        mean_high = np.mean(img[img >= t1])
        t_new = int(round((mean_low + mean_high) / 2))
    bin_img = np.zeros(np.shape(img))
    bin_img[img >= t1] = 255
    return t1, bin_img


img21 = cv2.imread('im1.bmp')
new_treshold21, new_img21 = iterative_threshold(img21)
print('Image "im1"')
print('New Treshold for "im1" is:{}', new_treshold21)

plt.figure()
plt.subplot2grid((2, 2), (1, 0), colspan=2), plt.hist(img21.ravel(), 255, [0, 255])
plt.subplot2grid((2, 2), (0, 0)), plt.imshow(img21, cmap='gray'), plt.title('Original Image')
plt.subplot2grid((2, 2), (0, 1)), plt.imshow(new_img21, cmap='gray'), plt.title('Binary Image')


# im3 - T_initial is set to be: (min_pix+max_pix)/2
def iterative_threshold_2(img):
    min_pix, max_pix = np.amin(img), np.amax(img)
    t_initial = int(round((min_pix + max_pix) / 2))
    t_new, t1 = t_initial, 0
    while t1 != t_new:
        t1 = t_new
        mean_low = np.mean(img[img < t1])
        mean_high = np.mean(img[img >= t1])
        t_new = int(round((mean_low + mean_high) / 2))
    bin_img = np.zeros(np.shape(img))
    bin_img[img >= t1] = 255
    return t1, bin_img


img23 = cv2.imread('im3.bmp')
new_treshold23_mean_init, new_img23a = iterative_threshold(img23)
new_treshold23_minmax_init, new_img23b = iterative_threshold_2(img23)
print('Image "im3"')
print('New Treshold for "im3" using the mean, is:{}', new_treshold23_mean_init)
print('New Treshold for "im3" using the min-max average, is:{}', new_treshold23_minmax_init)

plt.figure()
plt.subplot2grid((2, 2), (1, 0), colspan=2), plt.hist(img23.ravel(), 255, [0, 255])
plt.subplot2grid((2, 2), (0, 0)), plt.imshow(new_img23a, cmap='gray'), plt.title('T_intial=mean')
plt.subplot2grid((2, 2), (0, 1)), plt.imshow(new_img23b, cmap='gray'), plt.title('T_intial=0.5(max+min)')


img24 = cv2.imread('GaussianNoise.jpg')
new_treshold24, new_img24 = iterative_threshold(img24)
print('Our Image')
print('New treshold for our image using the mean, is:{}', new_treshold24)

plt.figure()
plt.subplot2grid((2, 2), (1, 0), colspan=2), plt.hist(img24.ravel(), 255, [0, 255])
plt.subplot2grid((2, 2), (0, 0)), plt.imshow(img24, cmap='gray'), plt.title('Original Image')
plt.subplot2grid((2, 2), (0, 1)), plt.imshow(new_img24, cmap='gray'), plt.title('Binary Image')
plt.show()

# Q 3:
img3 = cv2.imread('T.bmp', flags=0)

# Global Treashold
new_treshold3_glob, new_img3_glob = iterative_threshold(img3)

# Local Threshol
new_img3_local = np.zeros(img3.shape, np.uint8)
var_mat = np.ones((5, 7))
expand_var_mat = np.zeros(img3.shape, np.uint8)
for i in range(5):
    for j in range(7):
        fi, fj = i * 60, j * 60
        li, lj = (i + 1) * 60, (j + 1) * 60
        var_current_img = np.var(img3[fi:li, fj:lj])
        var_mat[i, j] = int(var_current_img)
        expand_var_mat[fi:li, fj:lj] = int(var_current_img)
expand_var_mat = expand_var_mat.reshape(expand_var_mat.shape[0:2])
var_treshold = 0.25*(var_mat.max()-var_mat.min())

var_above_T = []
for i in range(5):
    for j in range(7):
        fi, fj = i * 60, j * 60
        li, lj = (i + 1) * 60, (j + 1) * 60
        im_cropped = img3[fi:li, fj:lj]
        var_current_img = np.var(im_cropped)
        if var_current_img > var_treshold:
            new_treshold, new_img = iterative_threshold(im_cropped)
            var_above_T.append(new_treshold)
            new_img3_local[fi:li, fj:lj] = new_img

for i in range(5):
    for j in range(7):
        fi, fj = i * 60, j * 60
        li, lj = (i + 1) * 60, (j + 1) * 60
        im_cropped = img3[fi:li, fj:lj]
        var_current_img = np.var(im_cropped)
        if var_current_img <= var_treshold:
            binary_img = np.zeros(np.shape(im_cropped))
            binary_img[im_cropped > min(var_above_T)] = 255
            new_img3_local[fi:li, fj:lj] = binary_img

#  Var map and picture plots
plt.figure()
plt.imshow(expand_var_mat, cmap='gray'), plt.title('var map')

plt.figure()
plt.subplot2grid((1, 3), (0, 0)), plt.imshow(img3, cmap='gray'), plt.title('Original Image')
plt.xticks([]), plt.yticks([])
plt.subplot2grid((1, 3), (0, 1)), plt.imshow(new_img3_glob, cmap='gray'), plt.title('Global Treshold')
plt.xticks([]), plt.yticks([])
plt.subplot2grid((1, 3), (0, 2)), plt.imshow(new_img3_local, cmap='gray'), plt.title('Local Treshold')
plt.xticks([]), plt.yticks([])
plt.show()
