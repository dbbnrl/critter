# import the necessary packages
from sklearn.preprocessing import LabelEncoder
# DEPRECATED for model_selection: from sklearn.cross_validation import train_test_split
from keras.models import Sequential
from keras.layers import Activation
from keras.optimizers import SGD
from keras.layers import Convolution2D, MaxPooling2D, Dense, Flatten, Dropout
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
from imutils import paths
import numpy as np
import argparse
import cv2
import os

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
	help="path to input dataset")
args = vars(ap.parse_args())
 
# define the architecture of the network
model = Sequential()

model.add(Convolution2D(32, 3, 3, activation="relu", input_shape=(150, 150, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Convolution2D(32, 3, 3, activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Convolution2D(64, 3, 3, activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(1, activation="sigmoid"))

train_datagen = ImageDataGenerator(
        rescale=1.0/255,
        horizontal_flip=True)

train_generator = train_datagen.flow_from_directory(
        args["dataset"],
        follow_links=True,
        target_size=(150,150),
        color_mode='grayscale',
        batch_size=32,
        class_mode='binary')

valid_datagen = ImageDataGenerator(rescale=1.0/255)

valid_generator = valid_datagen.flow_from_directory(
        args["dataset"],
        follow_links=True,
        target_size=(150,150),
        color_mode='grayscale',
        batch_size=32,
        class_mode='binary')

# partition the data into training and testing splits, using 75%
# of the data for training and the remaining 25% for testing
# print("[INFO] constructing training/testing split...")
# (trainData, testData, trainLabels, testLabels) = train_test_split(
	# data, labels, test_size=0.25, random_state=42)

# train the model using SGD
print("[INFO] compiling model...")
model.compile(
        loss="binary_crossentropy",
        optimizer='rmsprop',
	metrics=["accuracy"])

print(model.summary())

model.fit_generator(
        train_generator,
        samples_per_epoch=2048,
        nb_epoch=50)
        # validation_data=validation_generator,
        # nb_val_samples=800)

# show the accuracy on the testing set
# print("[INFO] evaluating on testing set...")
# (loss, accuracy) = model.evaluate(testData, testLabels,
	# batch_size=128, verbose=1)
# print("[INFO] loss={:.4f}, accuracy: {:.4f}%".format(loss,
	# accuracy * 100))

