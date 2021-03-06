# config utf-8

import tensorflow as tf
import numpy as np
import csv


import PIL.Image
from io import BytesIO
from IPython.display import Image, display

def Fractal(a, filename, fmt='png'):
    a_cyclic = (6.28*a/20.0).reshape(list(a.shape)+[1])
    img = np.concatenate([10+20*np.cos(a_cyclic)
                        , 30+50*np.sin(a_cyclic)
                        , 155-80*np.cos(a_cyclic)], 2)

    img[a==a.max()] = 0

    a = img
    a = np.uint8(np.clip(a, 0, 255))
    f = BytesIO()
    pil_img = PIL.Image.fromarray(a)
    pil_img.save(filename, fmt)


if __name__ == '__main__':
    image_resolution = 1000

    ranges = open('range.csv', 'r')
    reader = csv.reader(ranges)
    header = next(reader)
    for l, line in enumerate(reader):
        line = map(float, line)
        x1, x2, y1, y2 = line
        range_x = abs(x1-x2) / image_resolution
        range_y = abs(y1-y2) / image_resolution
        y, x = np.mgrid[y1:y2:range_y, x1:x2:range_x]

        #y, x = np.mgrid[-1.3:1.3:0.005, -2:1:0.005]
        #y, x = np.mgrid[-1.2:-0.6:0.001, -0.35:0.05:0.001]
    # z = x + yi
        z = x + 1j * y

        xs = tf.constant(z.astype(np.complex64))
        zs = tf.Variable(xs)
        ns = tf.Variable(tf.zeros_like(xs, tf.float32))

    # Session
        sess = tf.Session()
        init_op = tf.global_variables_initializer()

        sess.run(init_op)

    # opetarion
    # z^2 + x
        zs_ = zs*zs + xs
        not_diverged = tf.abs(zs_) < 4

    # assign multiple operations
    # xs = zs_
    # ns = notdiverged
        step_op = tf.group(
            zs.assign(zs_),
            ns.assign_add(tf.cast(not_diverged, tf.float32)))

        for i in range(200):
            sess.run(step_op)

        Fractal(ns.eval(sess), 'test' + str(l) + '.png')

        sess.close()
    ranges.close()
