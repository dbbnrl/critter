#python3
# import the necessary packages
# from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential, load_model, save_model
from keras.layers import Activation
from keras.optimizers import SGD
from keras.layers import Convolution2D, MaxPooling2D, Dense, Flatten, Dropout
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator, Unbatch
from imutils import paths
import numpy as np
import matplotlib
from matplotlib import pyplot
from scipy.misc import toimage
import argparse
import cv2
import os

from dataprep import prep_data

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

model_file = "crapnet.h5"

classes = ['yes', 'no']

config=[
    ('data/clear', 'yes', 0.5),
    ('data/nonempty', 'no', 0.45),
    ('data/empty', 'no', 0.05),
    ]

img_size = (240, 320)
nb_epoch = 50

#construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
# ap.add_argument("-d", "--dataset", required=True,
	# help="path to input dataset")
ap.add_argument("-l", "--load-model", action='store_true')
ap.add_argument("-t", "--train", action='store_true')
ap.add_argument("-st", "--show-training", action='store_true')
ap.add_argument("-sv", "--show-validation", action='store_true')
ap.add_argument("-c", "--clock-prediction", action='store_true')
args = ap.parse_args()
 
if args.load_model:
    model = load_model(model_file)
else:
    # define the architecture of the network
    model = Sequential()

    model.add(Convolution2D(32, 3, 3, activation="relu", input_shape=(img_size + (1,))))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Convolution2D(32, 3, 3, activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Convolution2D(32, 3, 3, activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Convolution2D(64, 3, 3, activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(64, activation="relu"))
    model.add(Dropout(0.25))
    model.add(Dense(1, activation="sigmoid"))

    # train the model using SGD
    print("[INFO] compiling model...")
    model.compile(
            loss="binary_crossentropy",
            optimizer='nadam',
        metrics=["accuracy"])

print(model.summary())

def preprocess(x):
    # if np.random.random() < 0.5:
    #     x = 255.0 - x
    x -= np.mean(x)
    # x /= np.std(x)
    if np.random.random() < 0.5:
        x *= -1.

    # for display only:
    # x /= 2.5
    # x += 0.5

    return x

imgen = ImageDataGenerator(
                           # rescale=1.0/255,
                           # samplewise_center=True,
                           rotation_range=5,
                           width_shift_range=0.1,
                           height_shift_range=0.1,
                           # shear_range=0.,
                           # zoom_range=0.2,
                           preprocessing_function=preprocess,
                           horizontal_flip=True)

(trainIt, testIt) = prep_data(imgen, imgen, config, classes, 0.2,
                              target_size=img_size,
                              color_mode='grayscale',
                              class_mode='binary',
                              batch_size=32)

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
        for i in range(0, 9):
        # for i in range(0, 8):
            if validate:
                (x, y, yp) = next(gen)
            else:
                (x, y) = next(gen)
            axs[i].imshow(np.squeeze(x), cmap='gray', norm=norm)
            if (y < 0.5):
                title = "Match"
            else:
                title = "No match"
            # title=str(np.mean(x))[:4]
            if validate and (y != yp):
                if (yp < 0.5):
                    axs[i].set_title("False POS", color="red")
                else:
                    axs[i].set_title("False NEG", color="red")
            else:
                axs[i].set_title(title, color="black")
        # x = preprocess(x)
        # axs[8].imshow(np.squeeze(x), cmap='gray', norm=norm)


        pyplot.waitforbuttonpress()

if args.show_training:
    show_training_images(Unbatch(trainIt))

# partition the data into training and testing splits, using 75%
# of the data for training and the remaining 25% for testing
# print("[INFO] constructing training/testing split...")
# (trainData, testData, trainLabels, testLabels) = train_test_split(
	# data, labels, test_size=0.25, random_state=42)

if args.train:
    model.fit_generator(
            trainIt,
            samples_per_epoch=2016,
            nb_epoch=nb_epoch,
            validation_data=testIt,
            nb_val_samples=200,
            callbacks=[ModelCheckpoint(model_file, save_best_only=True)]
                       # TensorBoard(histogram_freq=1, write_graph=False, write_images=True)]
            )
    save_model(model, model_file)

# def accum_data(iter, nb_elem):
#     Xs = []
#     Ys = []
#     cnt = 0
#     while (cnt < nb_elem):
#         X, Y = next(iter)
#         Xs.append(X)
#         Ys.append(Y)
#         cnt = cnt + len(Y)
#     return (np.array(Xs), np.array(Ys))

class Comparator(object):
    def __init__(self, model, generator):
        self.model = model
        self.generator = generator
    def __iter__(self):
        return self
    def __next__(self):
        (X, Y) = next(self.generator)
        Yp = self.model.predict_classes(X)
        return (X, Y, Yp)

if args.show_validation:
    [loss, acc] = model.evaluate_generator(testIt, 200)
    print("LOSS =", loss, "ACC =", acc)
    show_training_images(Unbatch(Comparator(model, testIt)), validate=True)

if args.clock_prediction:
    gen = Unbatch(testIt)
    print("generating...")
    Xs = []
    for _ in range(1000):
        x, y = next(gen)
        Xs.append(x)
    model.predict_classes(np.array(Xs))

# show the accuracy on the testing set
# print("[INFO] evaluating on testing set...")
# (loss, accuracy) = model.evaluate(testData, testLabels,
	# batch_size=128, verbose=1)
# print("[INFO] loss={:.4f}, accuracy: {:.4f}%".format(loss,
	# accuracy * 100))

