from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf
import numpy as np
import sklearn
from sklearn import metrics
import pandas as pd
import json

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from time import time

# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS

#core params..
flags.DEFINE_string('model', 'log_reg', 'model names. See README for possible values.')  
flags.DEFINE_string('train_prefix', '/Users/april/Downloads/GraphSAGE_Benchmark-master/processed/10n_2a/cora/cora_50_b3/cora_unsupervised/unsup-embedding_pre/graphsage_mean_small_0.000010', 'prefix identifying training data. must be specified.')
flags.DEFINE_string('labels', '', 'path of class-map to be used.')

# Load data from path and split into train and test
def load_data(path, labels):
    
    # Get numpy arrays and column vectors of nodes
    npyEmbVal = np.load(path + '/val.npy')
    txtEmbVal = np.loadtxt(path + '/val.txt', dtype=int)[np.newaxis].T
    npyEmbTest = np.load(path + '/val-test.npy')
    txtEmbTest = np.loadtxt(path + '/val-test.txt')[np.newaxis].T
    
    # Initialize panda and sort sort sort by node
    embedsDf = pd.DataFrame(data=npyEmbVal[1:,1:],    # values
                             index=txtEmbVal[1:,0])    # 1st column as index
    embedsDf.sort_index(inplace=True)
    embedsTest = pd.DataFrame(data=npyEmbTest[1:,1:],    # values
                              index=txtEmbTest[1:,0])    # 1st column as index

    # Create column vector for labels
    class_map = json.load(open(labels))
    label_col = np.array([], dtype=int)
    for i in range(0, len(class_map)):
        if i in embedsDf.index:
            if class_map[str(i)] == [1, 0]:
                label_col = np.append(label_col, [0], axis=0)
            elif class_map[str(i)] == [0, 1]:
                label_col = np.append(label_col, [1], axis=0)
        else:
            print("Node not present: ",i)
    
    # Add label column to panda
    embedsDf.insert(0, "Labels", label_col.tolist(), True)
    return embedsDf


def split(df):
    # Split into data and labels
    feature_names = list(range(0, len(df.columns)-1))
    X = df.loc[:, feature_names]
    y = df['Labels']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1500, random_state=0)
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    return X_train, X_test, y_train, y_test


def train(train_data):
    X_train = train_data[0]
    X_test = train_data[1]
    y_train = train_data[2]
    y_test = train_data[3]
    
    t0 = time()
    t1 = 0
    if FLAGS.model == 'log_reg':
        logreg = LogisticRegression()
        logreg.fit(X_train, y_train)
        t1 = time()
        y_pred = logreg.predict(X_test)
        precision, recall, fscore, support = metrics.precision_recall_fscore_support(y_test, y_pred, average='binary')
        print('Accuracy of Logistic regression classifier on training set: {:.2f}'
             .format(logreg.score(X_train, y_train)))
        print('Test set:')
        print('\tPrecision: ', precision)
        print('\tRecall: {:.2f}', recall)
        print('\tF1: ', fscore)
    
    elif FLAGS.model == 'dec_tree':
        dtc = DecisionTreeClassifier()
        dtc.fit(X_train, y_train)
        t1 = time()
        y_pred = dtc.predict(X_test)
        precision, recall, fscore, support = metrics.precision_recall_fscore_support(y_test, y_pred, average='binary')
        print('Accuracy of Decision Tree classifier on training set: {:.2f}'
             .format(dtc.score(X_train, y_train)))
        print('Test set:')
        print('\tPrecision: ', precision)
        print('\tRecall: {:.2f}', recall)
        print('\tF1: ', fscore)
        
    elif FLAGS.model == 'knn':
        knn = KNeighborsClassifier()
        knn.fit(X_train, y_train)
        t1 = time()
        y_pred = knn.predict(X_test)
        precision, recall, fscore, support = metrics.precision_recall_fscore_support(y_test, y_pred, average='binary')
        print('Accuracy of K-NN classifier on training set: {:.2f}'
             .format(knn.score(X_train, y_train)))
        print('Test set:')
        print('\tPrecision: ', precision)
        print('\tRecall: {:.2f}', recall)
        print('\tF1: ', fscore)
        
    elif FLAGS.model == 'mlp':
        mlp = MLPClassifier()
        mlp.fit(X_train, y_train)
        t1 = time()
        y_pred = mlp.predict(X_test)
        precision, recall, fscore, support = metrics.precision_recall_fscore_support(y_test, y_pred, average='binary')
        print('Accuracy of MLP classifier on training set: {:.2f}'
             .format(mlp.score(X_train, y_train)))
        print('Test set:')
        print('\tPrecision: ', precision)
        print('\tRecall: {:.2f}', recall)
        print('\tF1: ', fscore)
        
    trainTime = t1 - t0
    print("Training time: ", trainTime)


def main(argv=None):
    print("Loading training data..")
    train_data = load_data(FLAGS.train_prefix, FLAGS.labels)
    
    split_data = split(train_data)
    
    print("Done loading training data..")
    train(split_data)


if __name__ == '__main__':
    tf.app.run()
