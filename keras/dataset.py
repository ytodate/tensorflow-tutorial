# config utf-8

import csv
import numpy as np

from keras.preprocessing.image import array_to_img, img_to_array, load_img
from keras.models import Sequential
from keras.layers import Dense, Activation

TRAIN = "train.csv"
#pil_img = load_img('data/train/cat_00000.png')
#np_img = img_to_array(pil_img)
#np_img = np_img.reshape((1,) + np_img.shape)

def read_image():
    return image

def load_data():
    num_data = sum(1 for line in open(TRAIN))
    f = open(TRAIN, 'r')
    reader = csv.reader(f)
    images = []
    labels = []

    #x_train = np.zeros((num_data, 3, 32, 32), dtype='uint8')
    x_train = np.zeros((num_data, 32, 32, 3), dtype='uint8')
    y_train = np.zeros((num_data,), dtype='uint8')


    index = 0
    for image_path, label in reader:
        pil_img = load_img(image_path)
        pil_img = pil_img.resize((32,32))
        np_img = img_to_array(pil_img)

        # reshape color height width
        #np_img = np_img.reshape((1,) + (np_img.shape[2], np_img.shape[1], np_img.shape[0]))
        np_img = np_img.reshape((1,) + np_img.shape)
        x_train[index, :, :, :] = np_img
        y_train[index] = label
        index += 1

    print(type(x_train))

    return x_train, y_train

def build(source=None):
    datagen = ImageDataGenerator(rescale=1. / 255)
    data_generator = datagen.flow_from_directory(
                source,  # this is the target directory
                target_size=(150, 150),  # all images will be resized to 150x150
                batch_size=11,
                class_mode='sparse')
    class_dictionary = data_generator.class_indices
    return data_generator, class_dictionary

if __name__ == '__main__':
    load_data()
