import numpy as np

class NN:
    n = 0
    weights = [np.array([])]
    sigmas = []
    dsigmas = []
    A = [np.array([])]
    Z = [np.array([])]

    def __init__(self, n, sizes, sigmas, dsigmas):
        assert n == len(sizes) == len(sigmas) == len(dsigmas)
        self.n = n
        self.weights = []
        self.sigmas = sigmas
        self.dsigmas = dsigmas
        self.A = [np.zeros(i) for i in sizes]
        self.Z = [np.zeros(i) for i in sizes]

        for i in range(n - 1):
            self.weights.append(2 * np.random.random((sizes[i], sizes[i + 1])) - 1)

    def dsigma(self, n, x):
        return self.dsigmas[n](x)

    def sigma(self, n, x):
        return self.sigmas[n](x)

    def evaluate(self, x: np.array):
        self.Z[0] = x
        self.A[0] = self.sigma(0, self.Z[0])

        for i in range(len(self.weights)):
            self.Z[i + 1] = self.A[i].dot(self.weights[i])
            self.A[i + 1] = self.sigma(i + 1, self.Z[i + 1])

        return self.A[self.n - 1]

    def train(self, x, y_star, alpha):
        y = self.evaluate(x)

        diffs = []

        l = self.n - 1

        error = 2 * (y - y_star)
        delta = error * self.dsigma(l, self.Z[l])
        diff = np.outer(self.A[l - 1], delta)
        diffs.append((l - 1, diff * alpha))

        l -= 1

        while l > 0:
            delta = delta.dot(self.weights[l].T) * self.dsigma(l, self.Z[l])
            diff = np.outer(self.A[l - 1], delta)
            diffs.append((l - 1, diff * alpha))
            l -= 1

        for (i, d) in diffs:
            self.weights[i] -= d

        return np.mean((y - y_star) ** 2)
