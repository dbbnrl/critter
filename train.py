#python3
# import the necessary packages
# from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential, load_model, save_model, model_from_config
from keras.layers import Activation
from keras.optimizers import SGD
from keras.layers import Convolution2D, MaxPooling2D, Dense, Flatten, Dropout, AveragePooling2D
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator, Unbatch
from keras.backend.tensorflow_backend import _to_tensor, _EPSILON
from keras import backend as K
# from imutils import paths
import numpy as np
import matplotlib
from matplotlib import pyplot
from scipy.misc import toimage
import argparse
# import cv2
import os
import time
import tensorflow as tf
from tensorflow.python.training.training import write_graph
from tensorflow.python.framework.graph_util import convert_variables_to_constants
from tensorflow.python.tools.optimize_for_inference_lib import optimize_for_inference
# from tensorflow.tools.quantize_graph import GraphRewriter

from dataprep import prep_data

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

model_file = "crapnet.h5"

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
pos_weight = 5.0

#construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
# ap.add_argument("-d", "--dataset", required=True,
	# help="path to input dataset")
ap.add_argument("-l", "--load-model", action='store_true')
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

def weighted_binary_crossentropy(output, target, pos_weight, from_logits=False):
    '''Binary crossentropy between an output tensor and a target tensor.
    '''
    # Note: tf.nn.softmax_cross_entropy_with_logits
    # expects logits, Keras expects probabilities.
    if not from_logits:
        # transform back to logits
        epsilon = _to_tensor(_EPSILON, output.dtype.base_dtype)
        output = tf.clip_by_value(output, epsilon, 1 - epsilon)
        output = tf.log(output / (1 - output))
    return tf.nn.weighted_cross_entropy_with_logits(output, target, pos_weight)

def my_loss_fn(y_true, y_pred):
    return K.mean(weighted_binary_crossentropy(y_pred, y_true, pos_weight), axis=-1)

def conv(model, layers, dim=3, **kwargs):
    model.add(Convolution2D(layers, dim, dim,
                            activation="relu",
                            border_mode="same",
                            init="he_normal",
                            **kwargs))

def pool(model, dim=2, **kwargs):
    model.add(MaxPooling2D(pool_size=(dim, dim),
                           **kwargs))

def flatten(model, **kwargs):
    model.add(Flatten(**kwargs))

def dense(model, layers, **kwargs):
    model.add(Dense(layers,
                    activation="relu",
                    init='he_normal',
                    **kwargs))

def finaldense(model, **kwargs):
    model.add(Dense(1,
                    activation="sigmoid",
                    init='he_normal',
                    **kwargs))

def dropout(model, pct, **kwargs):
    model.add(Dropout(pct, **kwargs))

def bottleneck(model, layers):
    conv(model, layers/2, dim=1)
    conv(model, layers/2)
    conv(model, layers, dim=1)

def finalavg(model, **kwargs):
    # model.add(Convolution2D(1, 1, 1, activation='linear', init='he_normal'))
    # dim = model.output_shape[1]
    # model.add(AveragePooling2D((dim, dim), strides=(1, 1)))
    # flatten(model)
    # model.add(Activation('sigmoid'))
    model.add(Convolution2D(1, 1, 1, activation=None, init='he_normal'))
    dim = model.output_shape[1]
    print('dim is ', dim)
    model.add(MaxPooling2D((dim, dim)))
    flatten(model)
    model.add(Activation('sigmoid'))

if args.load_model:
    model = load_model(model_file, custom_objects={'my_loss_fn':my_loss_fn})
else:
    # define the architecture of the network
    model = Sequential()

    # conv(model, 16, input_shape=(img_size+ (1,)))
    # pool(model)
    # conv(model, 16)
    # pool(model)
    # conv(model, 16)
    # pool(model)
    # conv(model, 16)
    # pool(model)
    # flatten(model)
    # dense(model, 16)
    # finaldense(model)

    conv(model, 16, input_shape=(img_size+ (1,))) # 3x3
    conv(model, 16, subsample=(2, 2)) # +2=5x5,/2
    conv(model, 16) # +4=9x9
    conv(model, 32, subsample=(2, 2)) # +4=13x13,/2
    conv(model, 32) # +8=21x21
    conv(model, 32, subsample=(2, 2)) # +8=29x29,/2
    conv(model, 32) # +16=45x45
    conv(model, 64, subsample=(2, 2)) # +16=61x61, /2
    conv(model, 64) # +32=93x93
    conv(model, 64, subsample=(2, 2)) # +32=125x125, /2
    conv(model, 64) # +64=189x189
    conv(model, 64, subsample=(2, 2)) # +64=253x253, /2
    conv(model, 64) # +128=381x381
    conv(model, 64, dim=1) # 381x381
    finalavg(model)

    # train the model using SGD
    print("[INFO] compiling model...")
    model.compile(
            # loss="binary_crossentropy",
            loss=my_loss_fn,
            optimizer='adam',
        metrics=["binary_accuracy"])

def preprocess(x):
    # if np.random.random() < 0.5:
    #     x = 255.0 - x
    # x /= 255.0

    # x -= np.mean(x)
    # std = np.std(x)
    # if np.random.random() < 0.5:
    #     std *= -1.
    # x /= std
    # x.clip(-2.0, 2.0)

    x -= 127.5
    if np.random.random() < 0.5:
        x /= 8.
    else:
        x /= -8.

    return x

imgen = ImageDataGenerator(
                           # rescale=1.0/255,
                           # samplewise_center=True,
                           rotation_range=3,
                           width_shift_range=0.12,
                           height_shift_range=0.07,
                           # shear_range=0.,
                           # zoom_range=0.2,
                           preprocessing_function=preprocess,
                           horizontal_flip=True)

if load_data:
    (trainIt, testIt) = prep_data(imgen, imgen, config, classes, args.val_fraction,
                                target_size=img_size,
                                color_mode='grayscale',
                                class_mode='binary',
                                batch_size=32)

gsum = 0.
gvar = 0.
gcount = 0

def show_images(gen, compare=False, find_mismatch=False):
    first = True
    pyplot.ion()
    (fig, axs) = pyplot.subplots(3, 3)
    axs = axs.flatten()
    for ax in axs:
        ax.axis('off')
    pyplot.show()
    norm=None
    # norm=matplotlib.colors.Normalize()
    # norm=matplotlib.colors.NoNorm()
    while True:
        global gsum, gvar, gcount
        got_mismatch = False
        for i in range(0, 9):
        # for i in range(0, 8):
            if compare:
                (x, y, yp) = next(gen)
            else:
                (x, y) = next(gen)
                if first:
                    yp = y
            scount = 224*224
            ssum = x.sum()
            smean = ssum / scount
            svar = np.var(x)
            if gcount:
                gmean = gsum / gcount
                delta = smean - gmean
                m_a = gvar*(gcount-1)
                m_b = svar*(scount-1)
                M2 = m_a + m_b + delta**2 * gcount * scount / (gcount + scount)
            else:
                M2 = ssum
            gsum += ssum
            gcount += scount
            gvar = M2 / (gcount - 1)
            c = (y < 0.5)
            cp = (yp < 0.5)
            first = False
            axs[i].imshow(np.squeeze(x), cmap='gray', norm=norm)
            if c:
                title = "Match"
                color="green"
            else:
                title = "No match"
                color="black"
            # title=str(np.mean(x))[:4]
            if compare:
                score = int(100*(1.0-yp))
            else:
                score = int(100*(1.0-y))
            # mean = np.mean(x)
            # score=int(100.0*mean)
            if (c != cp):
                got_mismatch = True
                if compare:
                    color = "red"
                    if cp:
                        title = "False POS"
                    else:
                        title = "False NEG"
            title += " [" +str(score) + "]"
            axs[i].set_title(title, color=color)
        # x = preprocess(x)
        # axs[8].imshow(np.squeeze(x), cmap='gray', norm=norm)
        print("#={0}, mean={1}, std={2}".format(gcount, gsum/gcount, gvar))
        if got_mismatch or not find_mismatch:
            pyplot.waitforbuttonpress()

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

if args.show:
    if args.data:
        # Unlabeled data
        gen = ImageDataGenerator(preprocessing_function=preprocess).flow_from_directory(
                args.data,
                color_mode='grayscale',
                target_size=img_size,
                shuffle=False,
                classes=['.'],
                class_mode=None)
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

# if args.import:
#     K.set_learning_phase(0)

if args.export:
    # K.set_learning_phase(0)
    config = model.get_config()
    weights = model.get_weights()
    K.clear_session()
    K.set_learning_phase(0)
    model = Sequential.from_config(config)
    # K.set_learning_phase(0)
    model.set_weights(weights)
    # saver = Saver()
    # saver.save(K.get_session(), "tf_checkpoint")
    graph_def = K.get_session().graph.as_graph_def()
    frozen_graph = convert_variables_to_constants(K.get_session(), graph_def, [model.output.name[:-2]])
    opt_graph = optimize_for_inference(frozen_graph, [model.input.name[:-2]], [model.output.name[:-2]], tf.float32.as_datatype_enum)
    tf.reset_default_graph()
    tf.import_graph_def(opt_graph, name="")
    # rewrite = GraphRewriter()
    write_graph(opt_graph, "./", args.export, as_text=False)
    print([o.name for o in tf.get_default_graph().get_operations()])
    # with open("tfnet.pb", "w") as f:
        # 1
        # f.write(str(graph_def))
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
