# from random import random
# from bisect import bisect
# import itertools
from keras.preprocessing.image import enumerate_images, FileListIterator
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import numpy as np

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
    x /= 48.


    return x

def preprocess_train(x):
    x = preprocess(x)
#    if np.random.random() < 0.5:
#        x *= -1.
    return x

train_gen = ImageDataGenerator(
                           # rescale=1.0/255,
                           # samplewise_center=True,
                           rotation_range=3,
                           width_shift_range=0.12,
                           height_shift_range=0.07,
                           # shear_range=0.,
                           # zoom_range=0.2,
                           preprocessing_function=preprocess_train,
                           horizontal_flip=True)

pred_gen = ImageDataGenerator(preprocessing_function=preprocess)

def prep_data(config, classes, test_size, batch_size=64, **kwargs):
        train_its = []
        test_its = []
        weights = []
        allX = []
        allY = np.empty(0)
        for (subdir, cls, weight) in config:
                print("Loading " + subdir)
                (nb_class, X, Y) = enumerate_images(subdir,
                                                    classes=classes,
                                                    dir_class=cls,
                                                    follow_links=True)
                allX += X
                allY = np.append(allY, Y)
                if test_size:
                        (Xtrain, Xtest, Ytrain, Ytest) = train_test_split(X, Y, test_size=test_size)
                else:
                        (Xtrain, Xtest, Ytrain, Ytest) = (X, X, Y, Y)
                train_its.append(FileListIterator(train_gen, Xtrain, Ytrain, nb_class,
                                                  dim_ordering=train_gen.dim_ordering,
                                                  batch_size=batch_size,
                                                  **kwargs))
                test_its.append(FileListIterator(train_gen, Xtest, Ytest, nb_class,
                                                  dim_ordering=train_gen.dim_ordering,
                                                  batch_size=batch_size,
                                                 **kwargs))
                weights.append(weight)
        trainIt = MergeIterator(train_its, weights=weights, batch_size=batch_size)
        testIt = MergeIterator(test_its, weights=weights, batch_size=batch_size)
        allIt = FileListIterator(pred_gen, allX, allY, nb_class,
                                 dim_ordering=pred_gen.dim_ordering,
                                 shuffle=False, batch_size=batch_size,
                                 **kwargs)
        return (trainIt, testIt, allIt)

def dir_gen(dirname, **kwargs):
        return pred_gen.flow_from_directory(
                        dirname,
                        shuffle=False,
                        classes=['.'],
                        class_mode=None,
                        **kwargs)

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
        Yp = self.model.predict_on_batch(X)
        return tup + (Yp,)

# Transform an iterator that yields (batch of X, batch of Y, ...) into one that
# yields individual (X, Y, ...) tuples
class Unbatch(object):
    def __init__(self, batchiter):
        self.batchiter = batchiter
        self.nextbatch()
    def __iter__(self):
        return self
    def reset(self):
        self.batchiter.reset()
    def nextbatch(self):
        tup = next(self.batchiter)
        if not isinstance(tup, tuple):
                tup = (tup,)
        self.iters = [iter(i) for i in tup]
    def __next__(self):
        try:
            vals = [next(i) for i in self.iters]
        except StopIteration:
            self.nextbatch()
            vals = [next(i) for i in self.iters]
        return tuple(vals)


# Create an iterator that draws randomly from the output of several sub-iterators.
# If weights is None, select evenly from among the sub-iterators.
# This iterator will pull from its sub-iterators in batches, and re-batch to its
# own batch size.
class MergeIterator(object):
        def __init__(self, subiters, batch_size=32, weights=None):
                self.nb_iters = len(subiters)
                self.subiters = [Unbatch(it) for it in subiters]
                self.batch_size = batch_size
                self.weights = weights
        def __iter__(self):
                return self
        def reset(self):
            for it in self.subiters:
                it.reset()
        def __next__(self):
                inds = np.random.choice(range(self.nb_iters),
                                        size=self.batch_size,
                                        p=self.weights)
                Xs = []
                Ys = []
                for i in inds:
                    X, Y = next(self.subiters[i])
                    Xs.append(X)
                    Ys.append(Y)
                return (np.array(Xs), np.array(Ys))

# class Buffered(object):
#         def __init__(self, subiter):
#                 self.subiter = subiter
#                 self.Xit = iter(())
#                 self.Yit = iter(())
#         def __iter__(self):
#                 return self
#         def __next__(self):
#                 try:
#                         return (next(self.Xit), next(self.Yit))
#                 except StopIteration:
#                         (X, Y) = next(self.subiter)
#                         self.Xit = iter(list(X))
#                         self.Yit = iter(list(Y))
#                         return (next(self.Xit), next(self.Yit))

# returns ([(X, Y)] per dir, (X, Y) test split)
# def load_and_split(config, test_size, **kwargs):
#         trains = []
#         Xtests = []
#         Ytests = []
#         for (dir, weight) in config:
#                 print("Loading " + dir)
#                 (X, Y) = load_all_images(dir, **kwargs)
#                 (Xtrain, Xtest, Ytrain, Ytest) = train_test_split(X, Y, test_size=test_size)
#                 trains.append((Xtrain, Ytrain))
#                 Xtests.extend(Xtest)
#                 Ytests.extend(Ytest)
#         return (trains, (np.array(Xtests), np.array(Ytests)))
