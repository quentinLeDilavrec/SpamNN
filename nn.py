import numpy as np

class NN:
    n = 0
    weights = [np.array([])]
    B = [np.array([])]
    sigmas = []
    dsigmas = []
    A = [np.array([])]
    Z = [np.array([])]

    def __init__(self, n, sizes, sigmas, dsigmas):
        assert n == len(sizes) == len(sigmas) == len(dsigmas)
        self.n = n
        self.weights = []
        self.B = []
        self.sigmas = sigmas
        self.dsigmas = dsigmas
        self.A = [np.zeros(i) for i in sizes]
        self.Z = [np.zeros(i) for i in sizes]
        
        self.count=0

        for i in range(n - 1):
            self.weights.append(2 * np.random.random((sizes[i], sizes[i + 1])) - 1)

        for i in range(n):
            self.B.append(2 * np.random.random(sizes[i]) - 1)

    def dsigma(self, n, x):
        return self.dsigmas[n](x)

    def sigma(self, n, x):
        return self.sigmas[n](x)

    def evaluate(self, x: np.array):
        self.Z[0] = x + self.B[0]
        self.A[0] = self.sigma(0, self.Z[0])

        for i in range(len(self.weights)):
            self.Z[i + 1] = self.A[i].dot(self.weights[i]) + self.B[i + 1]
            self.A[i + 1] = self.sigma(i + 1, self.Z[i + 1])

        return self.A[self.n - 1]

    def train(self, x, y_star, eta):
        y = self.evaluate(x)

        diffs = []

        l = self.n - 1

        error = 2 * (y - y_star)
        delta = error * self.dsigma(l, self.Z[l])
        diffs.append((l, delta))

        l -= 1

        while l >= 0:
            delta = delta.dot(self.weights[l].T) * self.dsigma(l, self.Z[l])
            diffs.append((l, delta))
            l -= 1

        for (i, d) in diffs:
            if i > 0:
                self.weights[i - 1] -= np.outer(self.A[i - 1], d) * eta
            self.B[i] -= d * eta
        
        self.count+=1

    def error(self, X, Y):
        return np.mean([(Y[i] - self.evaluate(X[i])) ** 2 for i in range(len(X))])

    def result_error(self, X, Y):
        return np.mean([(Y[i] - np.round(self.evaluate(X[i]))) ** 2 for i in range(len(X))])

    def get_training_count(self):
        return self.count