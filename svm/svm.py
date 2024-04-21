'''
Support Vector Machine Algorithm 
Implemeted for Wine-Quality Data Set
'''
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from ucimlrepo import fetch_ucirepo
import numpy as np


class SVM_HiperParams:
    def __init__(
        self, C=1.0, kernel="linear", learning_rate=0.01, epochs=500, rbf_sigma=0.1
    ) -> None:
        self.C = C
        self.kernel = kernel
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.sigma = rbf_sigma


class SVM:
    def __init__(self, params: SVM_HiperParams):
        self.params = params
        self.X = None
        self.y = None
        self.weights = None
        self.bias = 0

    def kernel(self, X1, X2):
        if self.params.kernel == "linear":
            return self.linear_kernel(X1, X2)
        else:
            return self.rbf_kernel(X1, X2)

    def linear_kernel(self, X1, X2):
        return np.dot(X1, X2.T)

    def rbf_kernel(self, X1, X2):
        return np.exp(-(1 / self.params.sigma**2)
            * np.linalg.norm(X1[:, np.newaxis] - X2[np.newaxis, :], axis=2) ** 2
        )

    def fit(self, X_features, y_targets):
        self.X = X_features
        self.y = y_targets
        self.losses = []

        self.weights = np.random.random(X_features.shape[0])
        self.bias = 0
        outer_sum = np.outer(y_targets, y_targets) * self.kernel(X_features, X_features)

        for _ in range(self.params.epochs):
            gradient = np.ones(X_features.shape[0]) - np.dot(outer_sum, self.weights)

            self.weights += self.params.learning_rate * gradient
            self.weights[self.weights > self.params.C] = self.params.C
            self.weights[self.weights < 0] = 0

            loss = np.sum(self.weights) - 0.5 * np.sum(np.outer(self.weights, self.weights) * outer_sum)
            self.losses.append(loss)

        id = np.where((self.weights) > 0 & (self.weights < self.params.C))[0]

        bias = y_targets[id] - (self.weights * y_targets).dot(
            self.kernel(X_features, X_features[id])
        )
        self.bias = np.mean(bias)

    def predict(self, X):
        return np.sign((self.weights * self.y).dot(self.kernel(self.X, X)) + self.bias)


class DataGenerator:
    def __init__(self, threshold=6) -> None:
        self.wine_quality = fetch_ucirepo(id=186)
        self.X = self.wine_quality.data.features
        self.y = self.wine_quality.data.targets
        self.y.loc[:, "quality"] = np.where(self.y["quality"] >= threshold, 1, -1)


def main():
    dataset = DataGenerator()
    X_train, X_test, y_train, y_test = train_test_split(
        np.array(dataset.X), 
        np.array(dataset.y), 
        test_size=0.2, 
        random_state=1
    )

    params = SVM_HiperParams(
        C=1, kernel="linear", 
        learning_rate=1e-12,
        epochs=500, 
        rbf_sigma=0.5
    )
    svm = SVM(params)

    test_C = [0.5, 1, 2]
    test_sigma = [0.1, 0.5, 2.5]
    test_lr = [1e-10, 1e-5, 0.01]

    # svm.fit(X_train, y_train.ravel())
    # test_prediction = svm.predict(X_test)
    # train_prediction = svm.predict(X_train)
    # test_accuracy =  accuracy_score(y_test.ravel(), test_prediction)
    # train_accuracy = accuracy_score(y_train.ravel(), train_prediction)

    # print("Linear kernel test sample accuracy:" , test_accuracy)
    # print('Linear kernel train sample accuracy:', train_accuracy)

    svm.params.kernel='rbf'

    for currentC in test_C:
        svm.params.C = currentC
        print(f'Current C: {currentC}')
        for sigma in test_sigma:
            print(f'    Current rbf_sigma: {sigma}')
            svm.params.sigma = sigma
            for learning_rate in test_lr:
                svm.params.learning_rate = learning_rate
                print(f'        Current learning rate: {learning_rate}')
                svm.fit(X_train, y_train.ravel())
                test_prediction = svm.predict(X_test)
                train_prediction = svm.predict(X_train)
                test_accuracy = accuracy_score(y_test.ravel(), test_prediction)
                train_accuracy = accuracy_score(y_train.ravel(), train_prediction)
                print("         Rbf kernel test sample accuracy:", test_accuracy)
                print('         Rbf kernel train sample accuracy:', train_accuracy)


if __name__ == "__main__":
    main()
