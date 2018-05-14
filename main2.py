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

with open('nn.pkl', 'rb') as f:
    nn = dill.load(f)

X = df_train
Y = df["spam_or_not"]

errors = [nn.error(X[:1000], Y[:1000])]
result_errors = [nn.result_error(X[:1000], Y[:1000])]
for i in range(1000, 100000):
    k = np.random.randint(0, 4600)
    nn.train(X[k], Y[k], 1000./(1000+nn.get_training_count()))
    if i % 10000 == 0:
        error = nn.error(X[:1000], Y[:1000])
        errors.append(error)
        print("Epoch: {}; Error: {}".format(i, error))
        result_errors.append(nn.result_error(X[:1000], Y[:1000]))


with open('nn.pkl', 'wb') as f:
    dill.dump(nn,f)
print(nn.error(X, Y))

plt.plot(errors)
plt.plot(result_errors, "r")
plt.ylim([0, 1])
plt.show()
