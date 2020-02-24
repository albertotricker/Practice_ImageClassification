from itertools import product
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv


class LBP:
    def __init__(self, window_size=(128, 64), cell_size=(16, 16), delta=(8, 8), num_nn=8, radius=1):
        self.window_size = window_size
        self.cell_size = cell_size
        self.delta = delta
        self.num_nn = num_nn
        self.radius = radius

    def compute_descriptor(self, img, method="basic"):
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

    def compute_imgLBP(self, img, method="basic"):
        img_x_size, img_y_size = img.shape
        imgLBP = np.zeros((img_x_size, img_y_size), dtype=int)
        pattern = ''
        pattern_x_index = np.round(-self.radius * np.sin(np.arange(0, 2 * np.pi, 2 * np.pi / self.num_nn))).astype(int)
        pattern_y_index = np.round(-self.radius * np.cos(np.arange(0, 2 * np.pi, 2 * np.pi / self.num_nn))).astype(int)
        for x, y in product(range(img_x_size), range(img_y_size)):
            for i in range(self.num_nn):
                nn_x_index = (x + pattern_x_index[i]) % img_x_size
                nn_y_index = (y + pattern_y_index[i]) % img_y_size
                if method == "basic":
                    pattern += str(int(img[nn_x_index, nn_y_index] >= img[x, y]))
            imgLBP[x, y] = int(pattern, 2)
            pattern = ''

        return imgLBP


if __name__ == "__main__":
    test_img = cv.imread("..//data/test/pedestrians/AnnotationsPos_0.000000_crop001002d_0.png")
    test_img = cv.cvtColor(test_img, cv.COLOR_BGR2GRAY)
    lbp = LBP()

    imgLBP = lbp.compute_imgLBP(test_img)
    plt.subplot(121)
    plt.imshow(test_img, cmap="gray")
    plt.title("Imagen original")
    plt.subplot(122)
    plt.imshow(imgLBP, cmap="gray")
    plt.title("Imagen LBP")
    plt.show()

    descriptor = lbp.compute_descriptor(test_img)
    counted_data = {i: descriptor[i] for i in range(len(descriptor))}
    val, weight = zip(*[(k, v) for k, v in counted_data.items()])
    plt.hist(val, weights=weight, bins=len(descriptor))
    plt.ylim((0, max(weight)))
    plt.show()
