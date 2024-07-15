import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from joblib import dump

def train_lin_simple(data, date):
    today = date.today()
    d1 = today.strftime("%d-%m-%Y")
    data_train, data_test, cluster_train, cluster_test = train_test_split(data[data.columns[:-1]], data['cluster'], random_state=0)
    clf = SVC(kernel='linear', C = 10E10, random_state=0, cache_size=4000)
    clf.fit(data_train, cluster_train)
    predictions = clf.predict(data_test) # TODO: Visualisierung hier wahrscheinlich entfernen
    cm = confusion_matrix(cluster_test, predictions, labels=clf.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf.classes_)
    disp.plot()
    plt.show()
    #dump(clf, "../models/simple_lin"+  d1 + ".pkl")
    return clf, data_train, data_test, cluster_train, cluster_test




def train_poly_simple(data, date):
    today = date.today()
    d1 = today.strftime("%d-%m-%Y")
    data_train, data_test, cluster_train, cluster_test = train_test_split(data[data.columns[:-1]], data['cluster'], random_state=0)
    clf = SVC(kernel='poly', degree=3, C = 10E10, random_state=0, cache_size=4000)
    clf.fit(data_train, cluster_train)
    predictions = clf.predict(data_test) # TODO: Visualisierung hier wahrscheinlich entfernen
    cm = confusion_matrix(cluster_test, predictions, labels=clf.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf.classes_)
    disp.plot()
    plt.show()
    dump(clf, "/home/jan/CAU/Masterarbeit/master-thesis-jan-bensien/Code/models/simple_poly"+  d1 + ".pkl")

def train_rbf_simple(data, date):
    today = date.today()
    d1 = today.strftime("%d-%m-%Y")
    data_train, data_test, cluster_train, cluster_test = train_test_split(data[data.columns[:-1]], data['cluster'], random_state=0)
    clf = SVC(kernel='rbf', C = 10E10, random_state=0, cache_size=4000)
    clf.fit(data_train, cluster_train)
    predictions = clf.predict(data_test) # TODO: Visualisierung hier wahrscheinlich entfernen
    cm = confusion_matrix(cluster_test, predictions, labels=clf.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf.classes_)
    disp.plot()
    plt.show()
    dump(clf, "/home/jan/CAU/Masterarbeit/master-thesis-jan-bensien/Code/models/simple_rbf"+   d1 + ".pkl")