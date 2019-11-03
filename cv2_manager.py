# FUNÇÕES MORFOLOGICAS
import numpy as np
import cv2
from skimage.io import imread, imsave


class CV2Manager:

    def __init__(self, plot_folder="Plots/rotation-translation/", files_folder='Datasets/Binary/'):
        self.plot_folder = plot_folder
        self.files_folder = files_folder

    def load_image(self, filename):
        self.filename = filename.split('.')[0] + '.png'
        self.img = imread(self.files_folder+filename)

    def rotate_and_translate_image(self):
        largura, altura = self.img.shape[:2]
        ponto = (altura / 2, largura / 2)  # ponto no centro da figura

        rotacao1 = cv2.getRotationMatrix2D(ponto, 90, 1.0)
        rotacionado1 = cv2.warpAffine(self.img, rotacao1, (largura, altura))
        imsave(self.plot_folder+'rotation-90-'+self.filename, rotacionado1)

        rotacao2 = cv2.getRotationMatrix2D(ponto, 120, 1.0)
        rotacionado2 = cv2.warpAffine(self.img, rotacao2, (largura, altura))
        imsave(self.plot_folder+'rotation-120-'+self.filename, rotacionado2)

        translation_matrix = np.float32([[1, 0, 70], [0, 1, 110]])
        img_translation = cv2.warpAffine(self.img, translation_matrix, (largura, altura))
        imsave(self.plot_folder+'tranlation-'+self.filename, img_translation)

        rotacao1 = cv2.getRotationMatrix2D(ponto, 45, 0.5)
        rotacionado1 = cv2.warpAffine(self.img, rotacao1, (largura, altura))
        imsave(self.plot_folder+'rotation-45-tamanho-'+self.filename, rotacionado1)

images = [
    'apple-1.gif',
    'deer-16.gif',
]

manager = CV2Manager()
for filename in images:
    manager.load_image(filename)
    manager.rotate_and_translate_image()
