import numpy as np
import matplotlib.pyplot as plt
import cv2


def thresholding(img, th):
    ret_img = np.zeros(np.shape(img))
    ret_img[img >= th] = 255
    return ret_img


# Q4
th4 = 230
img4 = cv2.imread("isch_head.bmp", flags=0)[8:392, 6:317]  # only the non-frame pixels
img4 = thresholding(img4, th4)
plt.imshow(img4, cmap='gray')
plt.title("Threshold = 230 [gray level]")


def sobelX(img):
    """sobel filter in the X direction"""
    f = np.array([-1, 0, 1, -2, 0, 2, -1, 0, 1]).reshape([3, 3])
    return cv2.filter2D(img, cv2.CV_64F, f)


def sobelY(img):
    """sobel filter in the Y direction"""
    f = np.array([-1, -2, -1, 0, 0, 0, 1, 2, 1]).reshape([3, 3])
    return cv2.filter2D(img, cv2.CV_64F, f)


def sobelP45(img):
    """sobel filter in the +45 degree direction"""
    f = np.array([-2, -1, 0, -1, 0, 1, 0, 1, 2]).reshape([3, 3])
    return cv2.filter2D(img, cv2.CV_64F, f)


def sobelN45(img):
    """sobel filter in the -45 degree direction"""
    f = np.array([0, -1, -2, 1, 0, -1, 2, 1, 0]).reshape([3, 3])
    return cv2.filter2D(img, cv2.CV_64F, f)


def plot4(tit, args):
    plt.figure()
    plt.subplot(221)
    plt.imshow(args[0], cmap='gray')
    plt.title("x")
    plt.subplot(222)
    plt.imshow(args[1], cmap='gray')
    plt.title("y")
    plt.subplot(223)
    plt.imshow(args[2], cmap='gray')
    plt.title("+45")
    plt.subplot(224)
    plt.imshow(args[3], cmap='gray')
    plt.title("-45")
    plt.suptitle(tit)

    plt.figure()
    plt.imshow(sum(args)/len(args), cmap='gray')
    plt.title(tit+" sum")


def plot2(tit, args):
    plt.figure()
    plt.subplot(211)
    plt.imshow(args[0], cmap='gray')
    plt.title("x")
    plt.subplot(212)
    plt.imshow(args[1], cmap='gray')
    plt.title("y")
    plt.suptitle(tit)

    plt.figure()
    # tot = sum(0.25*[args[i] for i in len(args)])
    plt.imshow(sum(args)/len(args), cmap='gray')
    plt.title(tit+" sum")


# Q6
img6 = cv2.imread("Additional Pics/lines_for_q6.jpg", flags=0)
th6 = 10

# self-made sobel filters applied to the image:
my_sobels = [thresholding(func(img6), th6) for func in [sobelX, sobelY, sobelP45, sobelN45]]
plot4('self-made sobel filters', my_sobels)



# cv2 sobel filters:
sobx = thresholding(cv2.Sobel(img6, cv2.CV_64F, 1, 0, ksize=3), th6)
soby = thresholding(cv2.Sobel(img6, cv2.CV_64F, 0, 1, ksize=3), th6)
plot2('cv2 sobel filters', [sobx, soby])


img6b = cv2.imread("T.bmp", flags=0)


def lap_o_gau(img, sigma=1):
    return cv2.Laplacian(cv2.GaussianBlur(img, (5, 5), sigma), cv2.CV_64F)


def zero_cross_map(img):
    shape = img.shape
    zcm = np.zeros(shape)
    for x in range(shape[0] - 1):
        for y in range(shape[1] - 1):
            if np.sign(img[x, y]) * np.sign(img[x + 1, y]) < 0 or np.sign(img[x, y]) * np.sign(img[x, y + 1]) < 0:
                zcm[x, y] = 255
    return zcm


logim6 = lap_o_gau(img6b)
dlogim6 = zero_cross_map(logim6)
logim6s4 = lap_o_gau(img6b, 4)
dlogim6s4 = zero_cross_map(logim6s4)

plt.figure()
plt.imshow(dlogim6, cmap='gray')
plt.title("sigma = 1")

plt.figure()
plt.imshow(dlogim6s4, cmap='gray')
plt.title("sigma = 4")
