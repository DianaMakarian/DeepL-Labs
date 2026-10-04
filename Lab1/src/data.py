from sklearn import datasets
import numpy as np

def load_iris():
    iris = datasets.load_iris()
    # print(iris.DESCR)
    return iris.data.astype(np.float64), iris.target.astype(np.int64)

def stratified_split(X, y, train_per_class=35, seed=0):
    rng = np.random.default_rng(seed)
    train_indices, test_indices = [], []

    for c in np.unique(y):
        curr_class_objs = np.where(y == c)[0]

        shuffled_class_objs = rng.permutation(curr_class_objs)

        train_indices.extend(shuffled_class_objs[:train_per_class])
        test_indices.extend(shuffled_class_objs[train_per_class:])

    X_train, y_train = X[train_indices], y[train_indices]
    X_test, y_test = X[test_indices], y[test_indices]
    return X_train, y_train, X_test, y_test

def standardize(X_train, X_test):
    for i in range(X_train.shape[1]):
        feature = X_train[:,i]

        mean = feature.mean()
        std = feature.std(ddof=0)

        X_train[:, i] = (X_train[:,i]-mean)/std
        X_test[:, i] = (X_test[:,i]-mean)/std
    return X_train, X_test


def iris_data_preprocessing():
    X, y = load_iris()
    X_train, y_train, X_test, y_test = stratified_split(X, y)
    X_train, X_test = standardize(X_train, X_test)
    return X_train, y_train, X_test, y_test

if __name__== "__main__":
    X_train, _, _, _ = iris_data_preprocessing()
    print(X_train)
