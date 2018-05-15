import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from nn import NN

import dill

df = pd.read_csv("spam.txt")
df_train = df.drop("spam_or_not", 1).as_matrix()

df_train -= np.amin(df_train, 0)
df_train /= np.amax(df_train, 0)
df_train = 2 * df_train - 1

sigma = lambda x: 1. / (1. + np.exp(-x))
dsigma = lambda x: sigma(x) * (1 - sigma(x))

X = np.array([[0, 0, 1],
              [0, 1, 1],
              [1, 0, 1],
              [1, 1, 1]])

Y = np.array([0, 0, 1, 1])

nn = NN(5, [df_train.shape[1], 70, 70, 70, 1], 5 * [sigma], 5 * [dsigma])

X = df_train
Y = df["spam_or_not"]

def get_train_test_inds(y,train_proportion=0.7):
    '''Generates indices, making random stratified split into training set and testing sets
    with proportions train_proportion and (1-train_proportion) of initial sample.
    y is any iterable indicating classes of each observation in the sample.
    Initial proportions of classes inside training and 
    testing sets are preserved (stratified sampling).
    '''

    y=np.array(y)
    train_inds = np.zeros(len(y),dtype=bool)
    test_inds = np.zeros(len(y),dtype=bool)
    values = np.unique(y)
    for value in values:
        value_inds = np.nonzero(y==value)[0]
        np.random.shuffle(value_inds)
        n = int(train_proportion*len(value_inds))

        train_inds[value_inds[:n]]=True
        test_inds[value_inds[n:]]=True

    return train_inds,test_inds

train_inds,test_inds = get_train_test_inds(Y,train_proportion=0.8)


import random
import math
print(np.argwhere(np.array(train_inds)==True))
print(random.choice(np.argwhere(train_inds==True)))

errors = []
result_errors = []
for i in range(1000, 1000000):
    k = random.choice(np.argwhere(train_inds==True))[0]
    # print(2000. * math.exp(-nn.get_training_count()/1000000-100))
    # print(2000./(1000+nn.get_training_count()))
    nn.train(
        X[k,:], 
        Y[k], 
        20000./(1+nn.get_training_count()**0.125))
    if nn.get_training_count() % 5000 == 0:
        print(2000./(1000+nn.get_training_count()))
        error = nn.error(
            X[test_inds], 
            Y[test_inds])
        errors.append(error)
        print("Epoch: {}; Error: {}".format(nn.get_training_count(), error))
        result_errors.append(nn.result_error(X[test_inds], Y[test_inds]))

with open('nn.pkl', 'wb') as f:
    dill.dump(nn,f)

print(nn.error(X, Y))

plt.plot(errors)
plt.plot(result_errors, "r")
plt.ylim([0, 1])
plt.show()
