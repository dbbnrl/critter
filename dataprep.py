from random import random
from bisect import bisect
import itertools
from keras.preprocessing.image import enumerate_images, FileListIterator, MergeIterator
# from image import enumerate_images
from sklearn.model_selection import train_test_split
import numpy as np

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

def prep_data(train_gen, test_gen, config, classes, test_size, **kwargs):
        train_its = []
        test_its = []
        weights = []
        for (subdir, cls, weight) in config:
                print("Loading " + subdir)
                (nb_class, X, Y) = enumerate_images(subdir,
                                                    classes=classes,
                                                    dir_class=cls,
                                                    follow_links=True)
                (Xtrain, Xtest, Ytrain, Ytest) = train_test_split(X, Y, test_size=test_size)
                train_its.append(FileListIterator(train_gen, Xtrain, Ytrain, nb_class,
                                                  dim_ordering=train_gen.dim_ordering,
                                                  **kwargs))
                test_its.append(FileListIterator(test_gen, Xtest, Ytest, nb_class,
                                                  dim_ordering=test_gen.dim_ordering,
                                                 **kwargs))
                weights.append(weight)
        trainIt = MergeIterator(train_its, weights=weights)
        testIt = MergeIterator(test_its, weights=weights)
        return (trainIt, testIt)
