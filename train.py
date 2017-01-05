#!/usr/bin/env python3

from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint, TensorBoard
# from keras.utils import np_utils
import numpy as np
import argparse
# import os
import time
import tensorflow as tf

from visualize import show_images
from dataprep import prep_data, dir_gen, Comparator, Unbatch
from model import model_setup, model_export

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

def_model_file = "crapnet.h5"

classes = ['yes', 'no']

config=[
    ('data/clear', 'yes', 0.4),
    ('data/nonempty', 'no', 0.5),
    ('data/empty', 'no', 0.1),
    ]

#img_size = (240, 320)
#img_size = (308, 308)
img_size = (224, 224)
nb_epoch = 500

#construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
# ap.add_argument("-d", "--dataset", required=True,
	# help="path to input dataset")
ap.add_argument("-l", "--load", default=def_model_file)
ap.add_argument("-t", "--train", action='store_true')
ap.add_argument("-v", "--validate", action='store_true')
ap.add_argument("-s", "--show", action='store_true')
ap.add_argument("-p", "--predict", action='store_true')
ap.add_argument("-d", "--data")
ap.add_argument("-m", "--mismatch", action='store_true')
ap.add_argument("-c", "--clock", action='store_true')
ap.add_argument("-e", "--export")
ap.add_argument("-i", "--import")
ap.add_argument("-f", "--val-fraction", type=float, default=0.1)
args = ap.parse_args()
 
load_data = args.train or args.validate
if args.mismatch:
    args.show = True

model_file = args.load
model = model_setup(img_size, model_file)

if load_data:
    (trainIt, testIt) = prep_data(config, classes, args.val_fraction,
                                  target_size=img_size,
                                  color_mode='grayscale',
                                  class_mode='binary',
                                  batch_size=32)

if args.show:
    if args.data:
        # Unlabeled data
        gen = dir_gen(args.data,
                      color_mode='grayscale',
                      target_size=img_size)
    elif args.validate:
        gen = testIt
        [loss, acc] = model.evaluate_generator(gen, 200)
        print("LOSS =", loss, "ACC =", acc)
    else:
        gen = trainIt
    if args.predict:
        gen = Unbatch(Comparator(model, gen))
    else:
        gen = Unbatch(gen)
    # If we're doing predictions and working from LABELED data, we can compare predictions
    # and labels.
    do_compare = args.predict and not args.data
    show_images(gen, compare=do_compare, find_mismatch=args.mismatch)

print(model.summary())

if args.export:
    model_export(model, args.export)
    exit()

if args.train:
    model.fit_generator(
            trainIt,
            samples_per_epoch=2016,
            nb_epoch=nb_epoch,
            validation_data=testIt,
            nb_val_samples=100,
            pickle_safe=True,
            callbacks=[ModelCheckpoint(model_file, save_best_only=True)]
                       # TensorBoard(histogram_freq=1, write_graph=False, write_images=True)]
            )
    # save_model(model, model_file)

if args.clock:
    shape = (1,) + img_size + (1,)
    X = np.zeros(shape)
    print("Running predictions...")
    start = time.time()
    for _ in range(200):
        model.predict_on_batch(X)
    done = time.time()
    delta = done - start
    print("Took", delta / 200.0, "s per.")
