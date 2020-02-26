from itertools import product
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv


class LBP:
    def __init__(self, window_size=(128, 64), cell_size=(16, 16), delta=(8, 8), num_nn=8, radius=1):
        """
        :param window_size: tuple. Size of the window that move around the full image. The full image size has to be
        multiple of the window size.
        :param cell_size: tuble. Size of the cell that move around the window. The window size has to be multiple of
        the cell size.
        :param delta: tuple. Number of pixel that the cell size moves in each iteration.
        :param num_nn: integer. Number of nearest neighbour to compute.
        :param radius: integer. Distance from neighbour to the pixel.
        """
        self.window_size = window_size
        self.cell_size = cell_size
        self.delta = delta
        self.num_nn = num_nn
        self.radius = radius

    def compute(self, img, method="basic"):
        """
        :param img: numpy.array. Full image.
        :param method: string, Type of lbp, "basic" or "uniform".
        :return: descriptor of the image.
        """
        imgLBP = self.compute_imgLBP(img, method)
        img_x_size, img_y_size = img.shape
        LBPdescriptor = []
        for window_x, window_y in product(range(0, img_x_size, self.window_size[0]),
                                          range(0, img_y_size, self.window_size[1])):
            window = imgLBP[window_x:(window_x + self.window_size[0]), window_y:(window_y + self.window_size[1])]
            for cell_x, cell_y in product(range(0, self.window_size[0] - self.delta[0], self.delta[0]),
                                          range(0, self.window_size[1] - self.delta[1], self.delta[1])):
                cell = window[cell_x:(cell_x + self.cell_size[0]), cell_y:(cell_y + self.cell_size[1])]
                cellLBPdescriptor = np.histogram(cell, bins=2 ** self.num_nn)[0]
                LBPdescriptor.append(cellLBPdescriptor)
        return np.array(LBPdescriptor).reshape(-1, )

    # def compute_imgLBP(self, img, method="basic"):
    #     img_x_size, img_y_size = img.shape
    #     imgLBP = np.zeros((img_x_size, img_y_size), dtype=int)
    #     pattern = ''
    #     pattern_x_index = np.round(-self.radius * np.sin(np.arange(0, 2 * np.pi, 2 * np.pi / self.num_nn))).astype(int)
    #     pattern_y_index = np.round(-self.radius * np.cos(np.arange(0, 2 * np.pi, 2 * np.pi / self.num_nn))).astype(int)
    #     for x, y in product(range(img_x_size), range(img_y_size)):
    #         for i in range(self.num_nn):
    #             nn_x_index = (x + pattern_x_index[i]) % img_x_size
    #             nn_y_index = (y + pattern_y_index[i]) % img_y_size
    #             if method == "basic":
    #                 pattern += str(int(img[nn_x_index, nn_y_index] >= img[x, y]))
    #         imgLBP[x, y] = int(pattern, 2)
    #         pattern = ''
    #
    #     return imgLBP
    def compute_imgLBP(self, img, method="basic"):
        """
        :param img: numpy.array. Full image.
        :param method: string, Type of lbp, "basic" or "uniform".
        :return: image of the same size that "img", but each pixel contains the decimal number of the local binary
        pattern.
        """
        img_x_size, img_y_size = img.shape
        nn_x_shift = np.round(-self.radius * np.sin(np.arange(0, 2 * np.pi, 2 * np.pi / self.num_nn))).astype(int)
        nn_y_shift = np.round(-self.radius * np.cos(np.arange(0, 2 * np.pi, 2 * np.pi / self.num_nn))).astype(int)
        power = np.flip(2 ** np.arange(self.num_nn))
        imgLBP = np.zeros((img_x_size, img_y_size, self.num_nn), dtype=int)
        for i in range(self.num_nn):
            imgLBP[:, :, i] = (np.roll(img, [nn_x_shift[i], nn_y_shift[i]], axis=(0, 1)) >= img) * power[i]
        imgLBP = np.sum(imgLBP, 2)

        if method == "uniform":
            # Generating Look up table
            lut = np.zeros(2 ** self.num_nn)
            E = 0
            for n in range(len(lut)):
                pattern = bin(n)[2:].zfill(self.num_nn)
                U = 0
                bit = pattern[-1]
                for i in range(len(pattern)):
                    if bit != pattern[i]:
                        U += 1
                        bit = pattern[i]
                if U > 2:
                    lut[n] = -np.inf
                else:
                    lut[n] = E
                    E += 1
            lut[np.where(lut == -np.inf)[0]] = E

            imgLBP = lut[imgLBP]

        return imgLBP.astype(int)


if __name__ == "__main__":
    """
    The next code test the different functions written in this file.
    """
    # We use one single image from the test set
    test_img = cv.imread("..//data/test/pedestrians/AnnotationsPos_0.000000_crop001002d_0.png")
    test_img = cv.cvtColor(test_img, cv.COLOR_BGR2GRAY)
    lbp = LBP()

    # Testing compute_imgLBP() method
    imgLBP1 = lbp.compute_imgLBP(test_img)
    imgLBP2 = lbp.compute_imgLBP(test_img, method="uniform")
    plt.subplot(131)
    plt.imshow(test_img, cmap="gray")
    plt.title("Imagen original")
    plt.subplot(132)
    plt.imshow(imgLBP1, cmap="gray")
    plt.title("Imagen LBP basic")
    plt.subplot(133)
    plt.imshow(imgLBP2, cmap="gray")
    plt.title("Imagen LBP uniform")
    plt.show()

    # Testing compute_descriptor() method. Be patient, it takes too long.
    descriptor = lbp.compute(test_img)
    counted_data = {i: descriptor[i] for i in range(len(descriptor))}
    val, weight = zip(*[(k, v) for k, v in counted_data.items()])
    plt.hist(val, weights=weight, bins=len(descriptor))
    plt.ylim((0, max(weight)))
    plt.show()
