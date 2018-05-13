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

Y = np.array([0, 0, 1, 1])

nn = NN(2, [3, 1], 2 * [sigma], 2 * [dsigma])

errors = []
for i in range(20000):
    k = np.random.randint(0, 4)
    nn.train(X[k], Y[k], 0.1)
    error = nn.error(X, Y)
    errors.append(error)
    print("Epoch: {}; Error: {}".format(i, error))

print(nn.error(X, Y))

plt.plot(errors)
plt.ylim([0, 1])
plt.show()
