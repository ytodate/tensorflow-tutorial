# config utf-8

import csv
import numpy as np

from keras.preprocessing.image import array_to_img, img_to_array, load_img
from keras.models import Sequential
from keras.layers import Dense, Activation

TRAIN = "train.csv"
TEST = "test.csv"
#pil_img = load_img('data/train/cat_00000.png')
#np_img = img_to_array(pil_img)
#np_img = np_img.reshape((1,) + np_img.shape)

def read_image():
    return image

def read_csv(filename):
    num_data = sum(1 for line in open(filename))
    f = open(filename, 'r')
    reader = csv.reader(f)
    images = []
    labels = []

    #x_train = np.zeros((num_data, 3, 32, 32), dtype='uint8')
    x = np.zeros((num_data, 32, 32, 3), dtype='uint8')
    y = np.zeros((num_data,), dtype='uint8')

    index = 0
    for image_path, label in reader:
        pil_img = load_img(image_path)
        pil_img = pil_img.resize((32,32))
        np_img = img_to_array(pil_img)

        # reshape color height width
        np_img = np_img.reshape((1,) + np_img.shape)

        x[index, :, :, :] = np_img
        y[index] = label

        index += 1

    f.close()

    return (x, y)


def load_data():
    train_datasets = read_csv(TRAIN)
    test_datasets = read_csv(TEST)

    return train_datasets, test_datasets

if __name__ == '__main__':
    load_data()
