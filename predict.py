# from keras.models import Sequential, load_model, save_model, model_from_config
# from keras.layers import Activation
# from keras.optimizers import SGD
# from keras.layers import Convolution2D, MaxPooling2D, Dense, Flatten, Dropout
# from keras.callbacks import ModelCheckpoint, TensorBoard
# from keras.utils import np_utils
# from keras.preprocessing.image import ImageDataGenerator, Unbatch
# from keras import backend as K
# from imutils import paths
import numpy as np
# import matplotlib
# from matplotlib import pyplot
# from scipy.misc import toimage
import argparse
# import cv2
import os
import time
import tensorflow as tf
from PIL import Image

from dataprep import preprocess

img_size = (224, 224)
trials = 200

ap = argparse.ArgumentParser()
ap.add_argument("model")
args = ap.parse_args()

# model_name, _ = os.path.splitext(args.model)
# model_file = "tfmodel/"+model_name+".pb"
model_file = args.model

graph_def = tf.GraphDef()
with open(model_file, "rb") as gfile:
    graph_def.ParseFromString(gfile.read())
tf.import_graph_def(graph_def, name="")
shape = (1,) + (img_size) + (1,)
img = Image.open("test.jpg")
img = img.convert('L')
img = img.resize(img_size, Image.ANTIALIAS)
arr = np.asarray(img, dtype='float32')
arr = arr.reshape(shape)
with tf.Session() as sess:
    graph = sess.graph
    input = graph.get_tensor_by_name('convolution2d_input_1:0')
    output = graph.get_tensor_by_name('Sigmoid:0')
    start = time.time()
    for _ in range(trials):
        arr = preprocess(arr)
        feed = {input : arr}
        preds = sess.run(output,
                         feed_dict=feed)
    done = time.time()
    delta = done - start
    print("Took", delta / trials, "s per.")
    pred = preds[0]
    print("Prediction=", 1.0-pred)

