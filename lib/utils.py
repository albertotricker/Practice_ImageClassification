import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
import os

from itertools import compress
from lib.LBP import LBP
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels

class cross_validation:
    """
    This class is used to compute a parallel cross-validation with multiprocess package.
    """
    def __init__(self, model, data, labels, kf):
        """
        :param model: sklearn object, it has to have a fit and predict method.
        :param data: numpy array, full dataset.
        :param labels: numpy array, full labels.
        :param kf: sklearn kfold object.
        """
        self.model = model
        self.data = data
        self.labels = labels
        self.train_index = []
        self.test_index = []
        for train_index, test_index in kf.split(data):
            self.train_index.append(train_index)
            self.test_index.append(test_index)
    def compute(self, i):
        """
        :param i: integer, iteration of the kfold.
        :return: float, accuracy of prediction in the ith kfold.
        """
        kf_train, kf_test = self.data[self.train_index[i], :], self.data[self.test_index[i], :]
        kf_train_labels, kf_test_labels = self.labels[self.train_index[i]], self.labels[self.test_index[i]]
        self.model.fit(kf_train, kf_train_labels)
        prediction = self.model.predict(kf_test)
        acc = np.sum(np.equal(kf_test_labels, prediction)) / len(kf_test_labels)
        return acc


def read_data(path="..//data/train/", file_extension=".png", descriptor_type=None, lbp_method="basic"):
    """
    This function read all the images in the path folder. It also calculate the descriptor specified by the argument
    description_type.
    :param path: string, folder that contains all the data tha want to be read. It has to contain one directory for
    each of the different classes of the images.
    :param file_extension: string, e.g. ".png", ".jpg", etc.
    :param descriptor_type: string, descriptor used to describe the images, e.g. "hog", "lbp", etc.
    :param lbp_method: string,  if descriptor_type is "lbp", this arguments specifies if the lbp descriptor is basic
    or uniform.
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
                elif descriptor_type == "lbp":
                    lbp = LBP()
                    descriptor = lbp.compute(img, method=lbp_method)
                    data.append(descriptor)
                    data_labels.append(numeric_label)
                else:
                    data.append(cv.cvtColor(img, cv.COLOR_BGR2GRAY))
                    data_labels.append(numeric_label)
                counter_samples += 1
        print(counter_samples, "images read of class", label, "- numeric label:", numeric_label)
        counter_samples = 0
        numeric_label += 1
    print('')

    return np.array(data).squeeze(), np.array(data_labels)


def plot_confusion_matrix(y_true, y_pred, classes, title=None, cmap=plt.cm.Blues):
    """
    :param y_true: numpy array, true labels.
    :param y_pred: numpy array, predicted labels.
    :param classes: numpy array, name of the classes.
    :param title: string, title of the figure.
    :param cmap: colormap of the figure.
    :return:
    """
    if not title:
        title = 'Confusion matrix'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_true, y_pred)]
    cm_norm = (cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]) * 100

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd') + '\n' + format(cm_norm[i, j], '.2f') + '%',
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    """
    The next code test the different functions written in this file.
    """
    train, train_labels = read_data()
    print(train.shape)
    print(train_labels.shape)
