#!/usr/bin/env python3

from keras.callbacks import TensorBoard
from keras.preprocessing.image import load_img
# from keras.utils import np_utils
import numpy as np
import argparse
import os
import time
import tensorflow as tf

from visualize import show_images
from dataprep import prep_data, dir_gen, Comparator, Unbatch, preprocess
from model import model_setup, model_export, model_checkpoint
from filtvis import vis_optimage, vis_activations

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

classes = ['yes', 'no']

config=[
    ('data/clear', 'yes', 0.4),
    ('data/nonempty', 'no', 0.4),
    ('data/allin', 'no', 0.2),
    # ('data/empty', 'no', 0.1),
    ]

nb_epoch = 500

#construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
# ap.add_argument("-d", "--dataset", required=True,
	# help="path to input dataset")
ap.add_argument("model")
ap.add_argument("-t", "--train", action='store_true')
ap.add_argument("-v", "--validate", action='store_true')
ap.add_argument("-a", "--all-validate", action='store_true')
ap.add_argument("-s", "--show", action='store_true')
ap.add_argument("-p", "--predict", action='store_true')
ap.add_argument("-d", "--data")
ap.add_argument("-m", "--mismatch", action='store_true')
ap.add_argument("-c", "--clock", action='store_true')
ap.add_argument("-e", "--export_tf", action='store_true')
ap.add_argument("-i", "--import_tf")
ap.add_argument("-f", "--val-fraction", type=float, default=0.1)
ap.add_argument("-r", "--learn-rate", type=float, default=0.001)
ap.add_argument("-b", "--batch-size", type=int, default=64)
ap.add_argument("-l", "--layer")
args = ap.parse_args()
 
load_data = args.train or args.validate or args.all_validate
if args.mismatch or args.layer:
    args.show = True

model_name, _ = os.path.splitext(args.model)
model = model_setup(model_name, args.learn_rate)
img_size = model.input_shape[1:3]

if load_data:
    (trainIt, testIt, allIt) = prep_data(config, classes, args.val_fraction,
                                  target_size=img_size,
                                  color_mode='grayscale',
                                  class_mode='binary',
                                  batch_size=args.batch_size)

if args.all_validate:
    if args.import_tf:
        model_file = args.import_tf
        graph_def = tf.GraphDef()
        with open(model_file, "rb") as gfile:
            graph_def.ParseFromString(gfile.read())
        tf.reset_default_graph()
        tf.import_graph_def(graph_def, name="")
        with tf.Session() as sess:
            graph = sess.graph
            input = graph.get_tensor_by_name('input_1:0')
            output = graph.get_tensor_by_name('Sigmoid:0')
            cnt = 0
            corrects = 0
            for (X, Y) in allIt:
                feed = {input : X}
                preds = sess.run(output, feed_dict=feed)
                Yp = np.squeeze(preds)
                Yp = np.round(Yp)
                corrects += np.sum(Y == Yp)
                cnt += len(Y)
                print("total =", cnt, "correct =", corrects, "acc =", (corrects / cnt))
                if (cnt >= allIt.nb_sample):
                    break
    else:
        print("Wait a long time...")
        [loss, acc] = model.evaluate_generator(allIt, allIt.nb_sample, pickle_safe=True)
        print("LOSS =", loss, "ACC =", acc)


if args.show:
    gen = None
    if args.data:
        # Unlabeled data
        gen = dir_gen(args.data,
                      color_mode='grayscale',
                      target_size=img_size)
    elif args.validate:
        gen = testIt
        [loss, acc] = model.evaluate_generator(gen, 200, pickle_safe=True)
        print("LOSS =", loss, "ACC =", acc)
    elif load_data:
        gen = trainIt
    if gen:
        if args.predict:
            gen = Unbatch(Comparator(model, gen))
        else:
            gen = Unbatch(gen)
        print(np.shape(next(gen)))
    if args.layer:
        if gen:
            vis_activations(model, args.layer, gen)
        else:
            vis_optimage(model, args.layer)
        exit()
    # If we're doing predictions and working from LABELED data, we can compare predictions
    # and labels.
    do_compare = args.predict and not args.data
    show_images(gen, compare=do_compare, find_mismatch=args.mismatch)

print(model.summary())

if args.export_tf:
    model_export(model, model_name)
    exit()

if args.train:
    model.fit_generator(
            trainIt,
            samples_per_epoch=4096,
            nb_epoch=nb_epoch,
            validation_data=testIt,
            nb_val_samples=512,
            pickle_safe=True,
            callbacks=[model_checkpoint(model_name)]
                       # TensorBoard(histogram_freq=1, write_graph=False, write_images=True)]
            )
    # save_model(model, model_file)

if args.clock:
    shape = (1,) + img_size + (1,)
    X = np.zeros(shape)
    print("Running predictions...")
    start = time.time()
    for _ in range(200):
        X = preprocess(X)
        model.predict_on_batch(X)
    done = time.time()
    delta = done - start
    print("Took", delta / 200.0, "s per.")
