from sklearn import svm
from sklearn import neighbors
from sklearn.mixture import GaussianMixture
from sklearn import tree
from sklearn import metrics
from utils import accuracy2
from dataset import *
import time


# clfs = [svm.SVC, neighbors.KNeighborsClassifier, tree.DecisionTreeClassifier]
# clfs = [GaussianMixture]


# def main():
#     dataset = FeatureDatasetV1(2, 'features-20')
#     percent = 0.8
#     train_size = int(len(dataset)*percent)
#     train_x = dataset.data[:train_size].numpy()
#     train_y = dataset.labels[:train_size].squeeze(1).numpy()
#     test_x = dataset.data[train_size:].numpy()
#     test_y = dataset.labels[train_size:].squeeze(1).numpy()
#     for clf_cls in clfs:
#         clf = clf_cls()
#         clf.fit(train_x, train_y)
#         pred = clf.predict(test_x)
#         acc, far, mdr = accuracy2(pred, test_y)
#         print(f'acc: {acc}, far: {far}, mdr: {mdr}')
#         mat = metrics.confusion_matrix(test_y[:6000], pred[:6000])
#         print(mat)


def main():
    dataset = CSIDatasetV2(2, 'CSI')
    percent = 0.8
    train_size = int(len(dataset)*percent)
    train_x = dataset.data[:train_size].numpy()
    train_y = dataset.labels[:train_size].squeeze(1).numpy()
    test_x = dataset.data[train_size:].numpy()
    test_y = dataset.labels[train_size:].squeeze(1).numpy()
    
    gmm = GaussianMixture(n_components=2, random_state=0)
    gmm.fit(train_x[:100])
    start = time.time()
    pred = gmm.predict(test_x)
    end = time.time()
    speed = (end-start)/test_x.shape[0] * 1000
    print(f'execute time each CSI data = {speed} ms')

if __name__ == '__main__':
    main()


