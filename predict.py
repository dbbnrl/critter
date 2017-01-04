#python3
# import the necessary packages
# from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential, load_model, save_model, model_from_config
from keras.layers import Activation
from keras.optimizers import SGD
from keras.layers import Convolution2D, MaxPooling2D, Dense, Flatten, Dropout
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator, Unbatch
from keras import backend as K
from imutils import paths
import numpy as np
import matplotlib
from matplotlib import pyplot
from scipy.misc import toimage
import argparse
import cv2
import os
import time
from tensorflow.python.training.training import write_graph
from tensorflow.python.framework.graph_util import convert_variables_to_constants
import tensorflow as tf
import tensorflow.contrib.quantization
from PIL import Image

model_file = "crapnet.h5"

#img_size = (240, 320)
# img_size = (308, 308)
img_size = (224, 224)
nb_epoch = 50

#construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
# # ap.add_argument("-d", "--dataset", required=True,
# 	# help="path to input dataset")
# ap.add_argument("-l", "--load-model", action='store_true')
# ap.add_argument("-t", "--train", action='store_true')
# ap.add_argument("-st", "--show-training", action='store_true')
# ap.add_argument("-sv", "--show-validation", action='store_true')
# ap.add_argument("-sm", "--show-mispredict", action='store_true')
# ap.add_argument("-c", "--clock-prediction", action='store_true')
# ap.add_argument("-e", "--export", action='store_true')
# ap.add_argument("-p", "--predict")
ap.add_argument("-m", "--model")
args = ap.parse_args()
 
# if args.show_mispredict:
    # args.show_validation = True

# if args.export:
#     K.set_learning_phase(0)
#     config = model.get_config()
#     weights = model.get_weights()
#     # K.clear_session()
#     model = Sequential.from_config(config)
#     model.set_weights(weights)
#     # saver = Saver()
#     # saver.save(K.get_session(), "tf_checkpoint")
#     graph_def = K.get_session().graph.as_graph_def()
#     frozen_graph = convert_variables_to_constants(K.get_session(), graph_def, [model.output.name[:-2]])
#     write_graph(frozen_graph, "./", "tf_export.pb", as_text=False)
#     # with open("tfnet.pb", "w") as f:
#         # 1
#         # f.write(str(graph_def))
#     exit()

def preprocess(x):
    # if np.random.random() < 0.5:
    #     x = 255.0 - x
    x -= np.mean(x)
    x /= np.std(x)
    # if np.random.random() < 0.5:
    #     x *= -1.

    # for display only:
    # x /= 2.5
    # x += 0.5

    return x

def show_training_images(gen, validate=False):
    pyplot.ion()
    (fig, axs) = pyplot.subplots(3, 3)
    axs = axs.flatten()
    for ax in axs:
        ax.axis('off')
    pyplot.show()
    norm=matplotlib.colors.Normalize()
    # norm=matplotlib.colors.NoNorm()
    while True:
        got_mispredict = False
        for i in range(0, 9):
        # for i in range(0, 8):
            if validate:
                (x, y, yp) = next(gen)
                cp = (yp < 0.5)
            else:
                (x, y) = next(gen)
            c = (y < 0.5)
            axs[i].imshow(np.squeeze(x), cmap='gray', norm=norm)
            if c:
                title = "Match"
                color="green"
            else:
                title = "No match"
                color="black"
            score = int(100*(1.0-y))
            # title=str(np.mean(x))[:4]
            if validate:
                score = int(100*(1.0-yp))
                if (c != cp):
                    got_mispredict = True
                    color = "red"
                    if cp:
                        title = "False POS"
                    else:
                        title = "False NEG"
            title += " [" +str(score) + "]"
            axs[i].set_title(title, color=color)
        # x = preprocess(x)
        # axs[8].imshow(np.squeeze(x), cmap='gray', norm=norm)
        if got_mispredict or not args.show_mispredict:
            pyplot.waitforbuttonpress()

graph_def = tf.GraphDef()
with open(args.model, "rb") as gfile:
    graph_def.ParseFromString(gfile.read())
tf.import_graph_def(graph_def, name="")
shape = (1,) + (img_size) + (1,)
img = Image.open("test.jpg")
img = img.convert('L')
img = img.resize(img_size, Image.ANTIALIAS)
img = np.asarray(img, dtype='float32')
img = img.reshape(shape)
with tf.Session() as sess:
    graph = sess.graph
    input = graph.get_tensor_by_name('convolution2d_input_1:0')
    output = graph.get_tensor_by_name('Sigmoid:0')
    feed = {input : img}
    start = time.time()
    for _ in range(200):
        img = preprocess(img)
        pred = sess.run(output,
                        feed_dict=feed)
        pred = pred[0]
    done = time.time()
    delta = done - start
    print("Took", delta / 200.0, "s per.")

class Comparator(object):
    def __init__(self, model, generator):
        self.model = model
        self.generator = generator
    def __iter__(self):
        return self
    def __next__(self):
        tup = next(self.generator)
        if not isinstance(tup, tuple):
            tup = (tup,)
        X = tup[0]
        Yp = self.model.predict_proba(X)
        return tup + (Yp,)

# if args.clock_prediction:
#     shape = (1,) + img_size + (1,)
#     X = np.zeros(shape)
#     print("Running predictions...")
#     start = time.time()
#     for _ in range(100):
#         model.predict_on_batch(X)
#     done = time.time()
#     delta = done - start
#     print("Took", delta / 200.0, "s per.")

# if args.predict:
#     X = ImageDataGenerator(preprocessing_function=preprocess).flow_from_directory(
#             args.predict,
#             color_mode='grayscale',
#             target_size=img_size,
#             shuffle=False,
#             classes=['.'],
#             class_mode=None)
#     show_training_images(Unbatch(Comparator(model, X)))
