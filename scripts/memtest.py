import memory_profiler, sys
from mlwpy import *

def knn_memtest(train_ftrs, train_tgt, test_ftrs):
    knn   = neighbors.KNeighborsClassifier(n_neighbors=3)
    fit   = knn.fit(train_ftrs, train_tgt)
    preds = fit.predict(test_ftrs)

def nb_memtest(train_ftrs, train_tgt, test_ftrs):
    nb    = naive_bayes.GaussianNB()
    fit   = nb.fit(train_ftrs, train_tgt)
    preds = fit.predict(test_ftrs)


if __name__ == "__main__":
    method = sys.argv[1]
    test_function = {'nb' : nb_memtest,
                     'knn':knn_memtest}[method]

    iris = datasets.load_iris()
    (iris_train_ftrs, iris_test_ftrs,
     iris_train_tgt, iris_test_tgt) = skms.train_test_split(iris.data,
                                                            iris.target,
                                                            test_size=.25)
    tup = (iris_train_ftrs, iris_train_tgt, iris_test_ftrs)
    base = memory_profiler.memory_usage()[0]
    mu = memory_profiler.memory_usage((test_function,tup),
                                       max_usage=True)[0]
    print("{:<3}: ~{:.4f} MiB".format(method, mu-base))
