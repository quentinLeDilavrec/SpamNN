import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from nn import NN

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

Y = np.array([0, 1, 1, 0])

nn = NN(3, [57, 50, 1], 3 * [sigma], 3 * [dsigma])

errors = []
for i in range(100000):
    k = np.random.randint(0, 4601)
    error = nn.train(df_train[k], df["spam_or_not"][k], 1)
    errors.append(error)
    print("Epoch: {}; Error: {}".format(i, error))

plt.plot(errors)
plt.show()
