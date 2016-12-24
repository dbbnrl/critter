from random import random
from bisect import bisect
import itertools
from keras.preprocessing.image import DirectoryIterator, ImageDataGenerator
from sklearn.model_selection import train_test_split
import numpy as np

# copied from python 3.6:
def choices(population, weights=None, *, cum_weights=None, k=1):
        """Return a k sized list of population elements chosen with replacement.

        If the relative weights or cumulative weights are not specified,
        the selections are made with equal probability.

        """
        if cum_weights is None:
            if weights is None:
                _int = int
                total = len(population)
                return [population[_int(random() * total)] for i in range(k)]
            cum_weights = list(itertools.accumulate(weights))
        elif weights is not None:
            raise TypeError('Cannot specify both weights and cumulative weights')
        if len(cum_weights) != len(population):
            raise ValueError('The number of weights does not match the population')
        total = cum_weights[-1]
        return [population[bisect(cum_weights, random() * total)] for i in range(k)]

class FakeDataGen(object):
        def random_transform(self, x):
                return x
        def standardize(self, x):
                return x

class Buffered(object):
        def __init__(self, subiter):
                self.subiter = subiter
                self.Xit = iter(())
                self.Yit = iter(())
        def __iter__(self):
                return self
        def __next__(self):
                try:
                        return (next(self.Xit), next(self.Yit))
                except StopIteration:
                        (X, Y) = next(self.subiter)
                        self.Xit = iter(list(X))
                        self.Yit = iter(list(Y))
                        return (next(self.Xit), next(self.Yit))

# returns (images, classes) pair
def load_all_images(dir, **kwargs):
        gen = DirectoryIterator(dir, FakeDataGen(),
                                shuffle=False,
                                batch_size=1000000,
                                follow_links=True,
                                classes=['yes', 'no'],
                                class_mode='binary',
                                color_mode='grayscale',
                                **kwargs
                                )
        return next(gen)

# returns ([(X, Y)] per dir, (X, Y) test split)
def load_and_split(config, test_size, **kwargs):
        trains = []
        Xtests = []
        Ytests = []
        for (dir, weight) in config:
                print("Loading " + dir)
                (X, Y) = load_all_images(dir, **kwargs)
                (Xtrain, Xtest, Ytrain, Ytest) = train_test_split(X, Y, test_size=test_size)
                trains.append((Xtrain, Ytrain))
                Xtests.extend(Xtest)
                Ytests.extend(Ytest)
        return (trains, (np.array(Xtests), np.array(Ytests)))

class Weighted(object):
        def __init__(self, subiters, weights, batch_size):
                self.subiters = subiters
                self.cum_weights = list(itertools.accumulate(weights))
                self.batch_size = batch_size
        def __iter__(self):
                return self
        def __next__(self):
                its = choices(self.subiters, cum_weights=self.cum_weights, k=self.batch_size)
                Xs = []
                Ys = []
                for it in its:
                        (X, Y) = next(it)
                        Xs.append(X)
                        Ys.append(Y)
                return (np.array(Xs), np.array(Ys))

def make_weighted_generator(XYpairs, weights, batch_size, **kwargs):
        # weights = 1.0 / np.asarray(weights)
        # print("weights: ", weights)
        iters = [Buffered(ImageDataGenerator(**kwargs).flow(X, Y, batch_size))
                 for (X, Y) in XYpairs]
        print("got #iters:", len(iters))
        return Weighted(iters, weights, batch_size)
        # return iters[0]

def prep_data(config, target_size, test_size, batch_size, **kwargs):
        (trains, tests) = load_and_split(config, test_size, target_size=target_size)
        weights = [weight for (dir, weight) in config]
        train_gen = make_weighted_generator(trains, weights, batch_size, **kwargs)
        return (train_gen, tests)
