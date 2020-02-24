from itertools import compress
import numpy as np
import cv2 as cv
import os


def read_data(path="..//data/train/", file_extension=".png", descriptor_type=None):
    """
    :param path: folder that contains all the data tha want to be read. It has to contain one directory for each of the
    different classes of the images.
    :param file_extension: e.g. ".png", ".jpg", etc.
    :param descriptor_type: descriptor used to describe the images, e.g. "hog", "lbp", etc.
    :return: two numpy arrays, one containing the data and the other containing the class of the data.
    """
    data = []
    data_labels = []
    labels = os.listdir(path)
    labels = list(compress(labels, [not folder.startswith(".") for folder in labels]))

    numeric_label = 0
    counter_samples = 0
    for label in labels:
        for filename in os.listdir(path+label):
            if filename.endswith(file_extension):
                filename = path + label + "/" + filename
                img = cv.imread(filename)
                if descriptor_type == "hog":
                    hog = cv.HOGDescriptor()
                    descriptor = hog.compute(img)
                    data.append(descriptor)
                    data_labels.append(numeric_label)
                else:
                    data.append(cv.cvtColor(img, cv.COLOR_BGR2GRAY))
                    data_labels.append(numeric_label)
                counter_samples += 1
        print(counter_samples, "images read of class", label, "- numeric label:", numeric_label)
        counter_samples = 0
        numeric_label += 1

    return np.array(data).squeeze(), np.array(data_labels)


if __name__ == "__main__":
    """
    Test the different functions written in this file.
    """
    train, train_labels = read_data()
    print(train.shape)
    print(train_labels.shape)
