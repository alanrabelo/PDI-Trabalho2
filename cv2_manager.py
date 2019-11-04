# FUNÇÕES MORFOLOGICAS
import numpy as np
import cv2
from skimage.io import imread, imsave
import math
import matplotlib.pyplot as plt
import pywt
from skimage import morphology
from skimage.morphology import disk, square, cube, diamond, rectangle, ball, star

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

    def houghCircles(self, plot_folder='Plots/Hough/'):
        img = self.img
        cimg = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 20,
                                   param1=50, param2=30, minRadius=0, maxRadius=40)

        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            # draw the outer circle
            cv2.circle(cimg, (i[0], i[1]), i[2], (0, 255, 0), 2)
            # draw the center of the circle
            cv2.circle(cimg, (i[0], i[1]), 2, (0, 0, 255), 3)

        imsave(plot_folder+'hough-circles'+self.filename, cimg)

    def houghLines(self, plot_folder='Plots/Hough/'):
        dst = cv2.Canny(self.img, 50, 200, None, 3)

        # Copy edges to the images that will display the results in BGR
        cdst = cv2.cvtColor(dst, cv2.COLOR_GRAY2BGR)
        lines = cv2.HoughLines(dst, 1, np.pi / 180, 150, None, 0, 0)

        if lines is not None:
            for i in range(0, len(lines)):
                rho = lines[i][0][0]
                theta = lines[i][0][1]
                a = math.cos(theta)
                b = math.sin(theta)
                x0 = a * rho
                y0 = b * rho
                pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * (a)))
                pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * (a)))
                cv2.line(cdst, pt1, pt2, (0, 0, 255), 3, cv2.LINE_AA)

        imsave(plot_folder+'hough-lines'+self.filename, cdst)

    def wavelet(self, plot_folder='Plots/Haar/'):
        img_haar = pywt.dwt2(self.img, 'haar')
        cA, (cH, cV, cD) = img_haar

        plt.figure(figsize=(9,9))

        plt.subplot(221)
        plt.imshow(cA, 'gray')
        plt.title('Original')
        plt.subplot(222)
        plt.imshow(cH, 'gray')
        plt.title('Horizontais')
        plt.subplot(223)
        plt.imshow(cV, 'gray')
        plt.title('Verticais')
        plt.subplot(224)
        plt.imshow(cD, 'gray')
        plt.title('Diagonais')
        plt.savefig(plot_folder+'haar'+self.filename)

    def plotResult(self, result, title, folder):
        fig = plt.figure(figsize=(8, 3.5), dpi=150)
        a = fig.add_subplot(1,2,1)
        a.axis('off')
        plt.imshow(self.img, cmap=plt.get_cmap('gray'))
        a.set_title('Original')

        a = fig.add_subplot(1,2,2)
        a.axis('off')
        plt.imshow(result, cmap=plt.get_cmap('gray'))
        a.set_title(title)

        plt.savefig(folder+title+'-'+self.filename)

    def fourier(self, plot_folder='Plots/Fourier/'):
        img = self.img
        rows, cols = img.shape
        img_dft = np.fft.fft2(img)
        img_dft_shift = np.fft.fftshift(img_dft)
        img_dft_mag = np.abs(img_dft_shift)

        self.plotResult(np.log(img_dft_mag), 'Espectro em frequência', plot_folder)

        img_idft = np.fft.ifft2(img_dft)
        img_inversa = np.abs(img_idft)

        self.plotResult(img_inversa, 'Imagem após IDFT', plot_folder)

        # cv2.imwrite('./Results/{}-kmeans.png'.format(self.name_file), result)

    def getMarkers(self, m, n):
        markers = np.zeros([m, n])
        # Center
        m = int(m / 2)
        n = int(n / 2)
        markers[20:40, 20:40] = 200
        markers[m:m + 20, n:n + 20] = 100

        return markers

    def watershed(self, title):
        image = self.img
        image_ext = morphology.dilation(image, disk(5)) - image

        m, n = image.shape
        markers = self.getMarkers(m, n)
        ws = morphology.watershed(image_ext, markers)

        cv2.imwrite('./Results/{}-ws-external-dilation.png'.format(title), 255 - image_ext)
        cv2.imwrite('./Results/{}-ws-markers.png'.format(title), markers)
        cv2.imwrite('./Results/{}-ws.png'.format(title), ws)
        self.plotWatershed(image, 255 - image_ext, markers, ws)

    def plotWatershed(self, image, dilation, markers, watershed, plot_folder='Plots/Watershed/'):
        fig, (ax0, ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=4, figsize=(8, 2.5), sharex=True, sharey=True)

        ax0.imshow(image, cmap=plt.get_cmap('gray'))
        ax0.set_title('Original')
        ax0.axis('off')

        ax1.imshow(dilation, cmap=plt.get_cmap('gray'))
        ax1.set_title('External Dilation')
        ax1.axis('off')

        ax2.imshow(markers, cmap=plt.get_cmap('gray'))
        ax2.set_title('Markers')
        ax2.axis('off')

        ax3.imshow(watershed, cmap=plt.get_cmap('nipy_spectral'), interpolation='nearest')
        ax3.set_title('Watershed')
        ax3.axis('off')

        fig.tight_layout()
        plt.savefig(plot_folder+'watershed'+self.filename)

images = [
    'apple-1.gif',
    'deer-16.gif',
]
gs_images = [
    'lena_gray_512.tif',
    'woman_blonde.tif',
]

manager = CV2Manager(files_folder='Datasets/GrayScale/')
for filename in gs_images:
    manager.load_image(filename)
    # manager.rotate_and_translate_image()
    # manager.houghCircles()
    # manager.houghLines()
    # manager.rotate_and_translate_image()
    # manager.fourier()
    # manager.wavelet()
    manager.watershed(filename)
