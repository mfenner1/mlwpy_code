import timeit, sys, functools as ft
from mlwpy import *

def knn_timetest(train_ftrs, train_tgt, test_ftrs):
    knn   = neighbors.KNeighborsClassifier(n_neighbors=3)
    fit   = knn.fit(train_ftrs, train_tgt)
    preds = fit.predict(test_ftrs)

def nb_timetest(train_ftrs, train_tgt, test_ftrs):
    nb    = naive_bayes.GaussianNB()
    fit   = nb.fit(train_ftrs, train_tgt)
    preds = fit.predict(test_ftrs)


if __name__ == "__main__":
    method = sys.argv[1]
    test_function = {'nb' : nb_timetest,
                     'knn':knn_timetest}[method]

    iris = datasets.load_iris()
    (iris_train, iris_test,
     iris_train_tgt, iris_test_tgt) = skms.train_test_split(iris.data,
                                                            iris.target,
                                                            test_size=.25)
    tup = (iris_train_ftrs, iris_train_tgt, iris_test_ftrs)
    call = ft.partial(test_function, *tup)
    tu = min(timeit.Timer(call).repeat(repeat=3, number=100))
    print("{:<3}: ~{:.4f} sec".format(method, tu))
