# Plotting Library
import matplotlib.pyplot as plt

from scipy import misc
from skimage import data
from skimage.io import imread

# FUNÇÕES MORFOLOGICAS
from skimage import morphology
from skimage.morphology import square, rectangle, disk


class MorphologyManager:

    def load_images(self, folder):
        image_list = []
        for filename in glob.glob(folder + '*.gif'):
            im = imread(filename)
            image_list.append(im)

        self.image_list = []

    def dilate_image(self, structuring_elements, image):

        results = []
        for element in structuring_elements:
            result = morphology.dilation(image, element)
            results.append(result)

        fig = plt.figure(figsize=(10, 10))
        a = fig.add_subplot(2, 2, 1)
        plt.imshow(image, cmap=plt.cm.gray)
        a.set_title('Original')
        plt.axis('off')

        a = fig.add_subplot(2, 2, 2)
        plt.imshow(results[0], cmap=plt.cm.gray)
        a.set_title('Dilatação 1')
        plt.axis('off')

        a = fig.add_subplot(2, 2, 3)
        plt.imshow(results[1], cmap=plt.cm.gray)
        a.set_title('Dilatação 2')
        plt.axis('off')

        a = fig.add_subplot(2, 2, 4)
        plt.imshow(results[2], cmap=plt.cm.gray)
        a.set_title('Dilatação 3')
        plt.axis('off')

        plt.tight_layout()
        plt.show()

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
        plt.show()

    def erode_image(self, structuring_elements, image):

        results = []
        for element in structuring_elements:
            result = morphology.erosion(image, element)
            results.append(result)

        fig = plt.figure(figsize=(10, 10))
        a = fig.add_subplot(2, 2, 1)
        plt.imshow(image, cmap=plt.cm.gray)
        a.set_title('Original')
        plt.axis('off')

        a = fig.add_subplot(2, 2, 2)
        plt.imshow(results[0], cmap=plt.cm.gray)
        a.set_title('Erosão 1')
        plt.axis('off')

        a = fig.add_subplot(2, 2, 3)
        plt.imshow(results[1], cmap=plt.cm.gray)
        a.set_title('Erosão 2')
        plt.axis('off')

        a = fig.add_subplot(2, 2, 4)
        plt.imshow(results[2], cmap=plt.cm.gray)
        a.set_title('Erosão 3')
        plt.axis('off')

        plt.tight_layout()
        plt.show()


import glob
images = [
    'Datasets/Binary/deer-16.gif',
    'Datasets/Binary/apple-1.gif',
]
image_list = []
for filename in images: #assuming gif
# for filename in glob.glob('Datasets/Binary/*.gif'): #assuming gif
    im = imread(filename)
    # Inversão na imagem
    image_list.append(im)

structuring_elements = [
    disk(10), square(20), rectangle(5, 20)
]

manager = MorphologyManager()
manager.dilate_image(structuring_elements, image_list[0])
manager.erode_image(structuring_elements, image_list[0])
manager.plot_structuring_elements(structuring_elements)

# Exibe imagens

#
# eros_3 = morphology.erosion(folha, disk(3))
# eros_15 = morphology.erosion(folha, disk(15))
# eros_30 = morphology.erosion(folha, disk(30))
#
# # Exibe imagens
# fig = plt.figure(figsize=(100,100))
# a = fig.add_subplot(1,3,1)
# plt.imshow(eros_3, cmap=plt.cm.gray)
# a.set_title('Erosão (3)')
# plt.axis('off')
#
# a = fig.add_subplot(1,3,2)
# plt.imshow(eros_15, cmap=plt.cm.gray)
# a.set_title('Erosão (15)')
# plt.axis('off')
#
# a = fig.add_subplot(1,3,3)
# plt.imshow(eros_30, cmap=plt.cm.gray)
# a.set_title('Erosão (30)')
# plt.axis('off')
#
# plt.tight_layout()
# plt.show()
#
# open_3  = morphology.opening(folha, disk(3))
# open_15 = morphology.opening(folha, disk(15))
# open_30 = morphology.opening(folha, disk(30))
#
# # Exibe imagens
# fig = plt.figure(figsize=(100,100))
# a = fig.add_subplot(1,4,1)
# plt.imshow(folha, cmap=plt.cm.gray)
# a.set_title('Original')
# plt.axis('off')
#
# a = fig.add_subplot(1,4,2)
# plt.imshow(open_3, cmap=plt.cm.gray)
# a.set_title('Abert (3)')
# plt.axis('off')
#
# a = fig.add_subplot(1,4,3)
# plt.imshow(open_15, cmap=plt.cm.gray)
# a.set_title('Abert (15)')
# plt.axis('off')
#
# a = fig.add_subplot(1,4,4)
# plt.imshow(open_30, cmap=plt.cm.gray)
# a.set_title('Abert (30)')
# plt.axis('off')
#
# plt.tight_layout()
# plt.show()
#
# dif_3  = folha - open_3
# dif_15 = folha - open_15
# dif_30 = folha - open_30
#
# # Exibe imagens
# fig = plt.figure(figsize=(100,100))
# a = fig.add_subplot(1,4,1)
# plt.imshow(dif_3, cmap=plt.cm.gray)
# a.set_title('Original')
# plt.axis('off')
#
# a = fig.add_subplot(1,4,2)
# plt.imshow(dif_15, cmap=plt.cm.gray)
# a.set_title('Abert (3)')
# plt.axis('off')
#
# a = fig.add_subplot(1,4,3)
# plt.imshow(dif_30, cmap=plt.cm.gray)
# a.set_title('Abert (15)')
# plt.axis('off')
#
# plt.tight_layout()
# plt.show()
#
# file_name = 'Folhas/Arbutus_BIN_29.bmp'
# folha_2 = imread(file_name)
# folha_2 = folha_2.max() - folha_2
#
# file_name = 'Folhas/Magnolia_BIN_20.bmp'
# folha_3 = imread(file_name)
# folha_3 = folha_3.max() - folha_3
#
# dif_f2 = folha_2 - morphology.opening(folha_2, disk(15))
# dif_f3 = folha_3 - morphology.opening(folha_3, disk(15))
#
# # Exibe imagens
# fig = plt.figure(figsize=(100,100))
# a = fig.add_subplot(2,2,1)
# plt.imshow(folha_2, cmap=plt.cm.gray)
# a.set_title('Original')
# plt.axis('off')
#
# a = fig.add_subplot(2,2,2)
# plt.imshow(dif_f2, cmap=plt.cm.gray)
# a.set_title('Orig - Abert (15)')
# plt.axis('off')
#
# a = fig.add_subplot(2,2,3)
# plt.imshow(folha_3, cmap=plt.cm.gray)
# a.set_title('Original')
# plt.axis('off')
#
# a = fig.add_subplot(2,2,4)
# plt.imshow(dif_f3, cmap=plt.cm.gray)
# a.set_title('Orig - Abert (15)')
# plt.axis('off')
#
#
# plt.tight_layout()
# plt.show()