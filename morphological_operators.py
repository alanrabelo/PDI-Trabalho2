# Plotting Library
import matplotlib.pyplot as plt

from scipy import misc
from skimage import data
from skimage.io import imread, imsave

# FUNÇÕES MORFOLOGICAS
from skimage import morphology
from skimage.morphology import square, rectangle, disk
from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans
import cv2
import numpy as np

class MorphologyManager:

    def __init__(self, plot_folder="Plots/Erosion-Dilation/", files_folder='Datasets/Binary/'):
        self.plot_folder = plot_folder
        self.files_folder = files_folder

    def load_images(self):
        self.img = imread(self.files_folder+filename)
        self.filename = filename.split('.')[0] + '.png'

    def dilate_image(self, structuring_elements):

        results = []
        for element in structuring_elements:
            result = morphology.dilation(self.img, element)
            results.append(result)

        fig = plt.figure(figsize=(10, 3))
        a = fig.add_subplot(1, 4, 1)
        plt.imshow(self.img, cmap=plt.cm.gray)
        a.set_title('Original')
        plt.axis('off')

        a = fig.add_subplot(1, 4, 2)
        plt.imshow(results[0], cmap=plt.cm.gray)
        a.set_title('Dilatação 1')
        plt.axis('off')

        a = fig.add_subplot(1, 4, 3)
        plt.imshow(results[1], cmap=plt.cm.gray)
        a.set_title('Dilatação 2')
        plt.axis('off')

        a = fig.add_subplot(1, 4, 4)
        plt.imshow(results[2], cmap=plt.cm.gray)
        a.set_title('Dilatação 3')
        plt.axis('off')

        plt.tight_layout()
        plt.savefig(self.plot_folder + 'dilation-' + self.filename)

    def plot_structuring_elements(self, structuring_elements):

        fig = plt.figure(figsize=(10, 5))
        a = fig.add_subplot(1, 3, 1)
        plt.imshow(structuring_elements[0], cmap=plt.cm.gray)
        a.set_title('Circulo de raio 10px')
        plt.axis('off')

        a = fig.add_subplot(1, 3, 2)
        plt.imshow(structuring_elements[1], cmap=plt.cm.gray)
        a.set_title('Quadrado de lado 20px')
        plt.axis('off')

        a = fig.add_subplot(1, 3, 3)
        plt.imshow(structuring_elements[2], cmap=plt.cm.gray)
        a.set_title('Retangulo 20px x 5px')
        plt.axis('off')

        plt.tight_layout()
        plt.savefig(self.plot_folder + 'structuring-elements.png')

    def erode_image(self, structuring_elements):

        results = []
        for element in structuring_elements:
            result = morphology.erosion(self.img, element)
            results.append(result)

        fig = plt.figure(figsize=(10, 3))
        a = fig.add_subplot(1, 4, 1)
        plt.imshow(self.img, cmap=plt.cm.gray)
        a.set_title('Original')
        plt.axis('off')

        a = fig.add_subplot(1, 4, 2)
        plt.imshow(results[0], cmap=plt.cm.gray)
        a.set_title('Erosão 1')
        plt.axis('off')

        a = fig.add_subplot(1, 4, 3)
        plt.imshow(results[1], cmap=plt.cm.gray)
        a.set_title('Erosão 2')
        plt.axis('off')

        a = fig.add_subplot(1, 4, 4)
        plt.imshow(results[2], cmap=plt.cm.gray)
        a.set_title('Erosão 3')
        plt.axis('off')

        plt.tight_layout()
        plt.savefig(self.plot_folder +'erosion-' + self.filename)

    def morphological_gradient(self, plot_folder="Plots/4-Morphological-Gradient/"):

        cont1 = morphology.dilation(self.img, disk(3)) - morphology.erosion(self.img, disk(3))
        cont2 = morphology.dilation(self.img, disk(5)) - morphology.erosion(self.img, disk(5))
        cont3 = morphology.dilation(self.img, disk(10)) - morphology.erosion(self.img, disk(10))

        fig = plt.figure(figsize=(10, 3))
        a = fig.add_subplot(1, 4, 1)
        plt.imshow(self.img, cmap=plt.cm.gray)
        a.set_title('Original')
        plt.axis('off')

        a = fig.add_subplot(1, 4, 2)
        plt.imshow(cont1, cmap=plt.cm.gray)
        a.set_title('Raio 3')
        plt.axis('off')

        a = fig.add_subplot(1, 4, 3)
        plt.imshow(cont2, cmap=plt.cm.gray)
        a.set_title('Raio 5')
        plt.axis('off')

        a = fig.add_subplot(1, 4, 4)
        plt.imshow(cont3, cmap=plt.cm.gray)
        a.set_title('Raio 10')
        plt.axis('off')

        plt.tight_layout()
        plt.savefig(plot_folder +'morphological-gradient-' + self.filename)

    def kmeans(self, number_of_clusters=5, plot_folder="Plots/7-K-means-clustering/"):

        (h1, w1) = self.img.shape[:2]

        image = self.img.reshape((h1 * w1, 1))
        final_images = []

        for i in range(2, 7):

            clt = KMeans(n_clusters=i)

            labels = clt.fit_predict(image)
            quant = clt.cluster_centers_.astype("uint8")[labels]

            quant = quant.reshape((h1, w1))
            final_images.append(quant)

        fig = plt.figure(figsize=(10, 8))
        a = fig.add_subplot(2, 3, 1)
        plt.imshow(self.img, cmap=plt.cm.gray)
        a.set_title('Original')
        plt.axis('off')

        a = fig.add_subplot(2, 3, 2)
        plt.imshow(final_images[0], cmap=plt.cm.gray)
        a.set_title('k = 2')
        plt.axis('off')

        a = fig.add_subplot(2, 3, 3)
        plt.imshow(final_images[1], cmap=plt.cm.gray)
        a.set_title('k = 3')
        plt.axis('off')

        a = fig.add_subplot(2, 3, 4)
        plt.imshow(final_images[2], cmap=plt.cm.gray)
        a.set_title('k = 4')
        plt.axis('off')

        a = fig.add_subplot(2, 3, 5)
        plt.imshow(final_images[3], cmap=plt.cm.gray)
        a.set_title('k = 5')
        plt.axis('off')

        a = fig.add_subplot(2, 3, 6)
        plt.imshow(final_images[4], cmap=plt.cm.gray)
        a.set_title('k = 6')
        plt.axis('off')

        plt.tight_layout()
        plt.savefig(plot_folder + 'k-means-' + self.filename)

        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

    def detect_circles(self):

        img = cv2.medianBlur(self.img, 5)
        cimg = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)

        circles = cv2.HoughCircles(cimg, cv2.HOUGH_GRADIENT, 1, 20,
                                   param1=100, param2=120, minRadius=0, maxRadius=0)

        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            # draw the outer circle
            cv2.circle(cimg, (i[0], i[1]), i[2], (0, 255, 0), 2)
            # draw the center of the circle
            cv2.circle(cimg, (i[0], i[1]), 2, (0, 0, 255), 3)

        cv2.imshow('detected circles', cimg)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

import glob
bw_images = [
    'deer-16.gif',
    'apple-1.gif',
]

# gs_images = [
#     'lena_gray_512.tif',
#     'woman_blonde.tif',
# ]
gs_images = [
    'coins.tif',
]

# manager = MorphologyManager()
# structuring_elements = [
#     disk(10), square(20), rectangle(5, 20)
# ]
# manager.plot_structuring_elements(structuring_elements)
#
# for filename in bw_images:
#     manager.filename = filename
#     manager.load_images()
#     manager.dilate_image(structuring_elements)
#     manager.erode_image(structuring_elements)
#     # manager.morphological_gradient()


manager = MorphologyManager(files_folder='Datasets/GrayScale/')
structuring_elements = [
    disk(10), square(20), rectangle(5, 20)
]
manager.plot_structuring_elements(structuring_elements)

for filename in gs_images:
    manager.filename = filename
    manager.load_images()
    # manager.dilate_image(structuring_elements)
    # manager.erode_image(structuring_elements)
    # manager.morphological_gradient()
    #
    # for i in range(2, 6):
    #     manager.kmeans(i)
    manager.detect_circles()
