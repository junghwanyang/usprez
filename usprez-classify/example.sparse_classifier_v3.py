# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 12:04:31 2015

@author: arash
"""

import pandas as pd
#from scikits.statsmodels.tools import categorical
import numpy as np
from time import time

from sklearn import preprocessing


from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import RidgeClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.ensemble import RandomForestClassifier
from sklearn.lda import LDA
from sklearn.utils.extmath import density
from sklearn import metrics
from sklearn.cross_validation import train_test_split

#df = pd.read_csv('dtm_allwords_1643.csv')
df = pd.read_csv('ko.dtm.noname/ko.dtm.noname.csv')
df_varname = pd.read_csv('ko.dtm.noname/variable.names.csv',encoding='utf-8')
df_whole_dtm = pd.read_csv('ko.dtm.noname/ko.whole.dtm.noname.csv')


dtM = df.as_matrix()[:,2:]
feature_names = df_varname[u'x'].as_matrix()
dtM_whole =  df_whole_dtm.as_matrix()[:,1:]

ideology = df.as_matrix()[:,1]
le = preprocessing.LabelEncoder()
le.fit(ideology)
label = le.transform(ideology) 


#label = np.dot(categorical(ideology, drop=True),np.array([1,2,3]))

X_train, X_test, y_train, y_test = train_test_split( dtM, label, test_size=0.5, random_state=42)


#dataset_type = (df['Training_test']).as_matrix()
#split_dataset = np.dot(categorical(dataset_type, drop=True),np.array([1,2]))
#
#training_dataset = [dtM[split_dataset==2,:],label[split_dataset==2]]
#testing_dataset = [dtM[split_dataset==1,:],label[split_dataset==1]]

#X_train = training_dataset[0]
#y_train = training_dataset[1]
#
#X_test = testing_dataset[0]
#y_test = testing_dataset[1]

categories = [u'C',u'L',u'U', u'M' ]

category_wights = {1:(label==1).sum() , 2:(label==2).sum() , 3:(label==3).sum() }

#class_weight =  dict({1 : 1} , {2 : 1} , {3 : 1})
def trim(s):
    """Trim string to fit on terminal (assuming 80-column display)"""
    return s if len(s) <= 80 else s[:77] + "..."
    
    
###############################################################################
# Benchmark classifiers
def benchmark(clf, name):
    print('_' * 80)
    print("Training: ")
    print(clf)
    t0 = time()
    clf.fit(X_train, y_train)
    train_time = time() - t0
    print("train time: %0.3fs" % train_time)

    t0 = time()
    pred = clf.predict(X_test)
    test_time = time() - t0
    print("test time:  %0.3fs" % test_time)

    score = metrics.accuracy_score(y_test, pred)
    print("accuracy:   %0.3f" % score)

    if hasattr(clf, 'coef_'):
        print("dimensionality: %d" % clf.coef_.shape[1])
        print("density: %f" % density(clf.coef_))

#        if opts.print_top10 and feature_names is not None:
        print("top 10 keywords per class:")
        for i, category in enumerate(categories):
            top10 = np.argsort(clf.coef_[i])[-10:]
            print(trim("%s: %s"
                  % (category, " ".join(feature_names[top10]))))

    #if opts.print_report:
    print("classification report:")
    print(metrics.classification_report(y_test, pred,
                                            target_names=categories))

    #if opts.print_cm:
    print("confusion matrix:")
    print(metrics.confusion_matrix(y_test, pred))

    clf_descr = str(clf).split('(')[0]
    
    
    #predicted_labels = le.inverse_transform(clf.predict(dtM_whole))
    #np.savetxt(name+'.csv', predicted_labels, delimiter=",")

    return clf_descr, score, train_time, test_time


results = []
for clf, name in (
        (RidgeClassifier(alpha=1.0,tol=1e-2, solver="lsqr"), "Ridge Classifier"),
        #(RidgeClassifier(alpha=1.0,tol=1e-2, solver="lsqr",class_weight={1:100, 2:100, 3:1}), "Ridge Classifier"),
        #(RidgeClassifier(alpha=1.0,tol=1e-2, solver="lsqr",class_weight=category_wights), "Ridge Classifier"),
        (Perceptron(n_iter= 100), "Perceptron"),
        (PassiveAggressiveClassifier(n_iter=100), "Passive-Aggressive"),
        (KNeighborsClassifier(n_neighbors=10), "kNN"),
        (RandomForestClassifier(n_estimators=100), "Random forest"),
        (LDA(), "Linear Discriminant Analysis"),
        (LinearSVC(), "SVM")
        ):
    print('=' * 80)
    print(name)
    results.append(benchmark(clf, name))

#import csv
#rows = zip(results[0],results[1],results[2])
#with open('class_predictions.csv', 'wb') as f:
#    csv.writer(f).writerows(rows)





#
#from sklearn.feature_selection import RFECV
#from sklearn.svm import SVC
#
#estimator = SVC(kernel="linear")
#selector = RFECV(estimator, step=1)
#selector = selector.fit(X_train, y_train)
#selector.ranking_


#for penalty in ["l2", "l1"]:
#    print('=' * 80)
#    print("%s penalty" % penalty.upper())
#    # Train Liblinear model
#    results.append(benchmark(LinearSVC(loss='l2', penalty=penalty,
#                                            dual=False, tol=1e-3)))
#
#    # Train SGD model
#    results.append(benchmark(SGDClassifier(alpha=.0001, n_iter=50,
#                                           penalty=penalty)))
#
## Train SGD with Elastic Net penalty
#print('=' * 80)
#print("Elastic-Net penalty")
#results.append(benchmark(SGDClassifier(alpha=.0001, n_iter=50,
#                                       penalty="elasticnet")))
#
## Train NearestCentroid without threshold
#print('=' * 80)
#print("NearestCentroid (aka Rocchio classifier)")
#results.append(benchmark(NearestCentroid()))
#
## Train sparse Naive Bayes classifiers
#print('=' * 80)
#print("Naive Bayes")
#results.append(benchmark(MultinomialNB(alpha=.01)))
#results.append(benchmark(BernoulliNB(alpha=.01)))
#
#print('=' * 80)
#print("LinearSVC with L1-based feature selection")
## The smaller C, the stronger the regularization.
## The more regularization, the more sparsity.
#results.append(benchmark(Pipeline([
#  ('feature_selection', LinearSVC(penalty="l1", dual=False, tol=1e-3)),
#  ('classification', LinearSVC())
#])))
#
## make some plots
#
#indices = np.arange(len(results))
#
#results = [[x[i] for x in results] for i in range(4)]
#
#clf_names, score, training_time, test_time = results
#training_time = np.array(training_time) / np.max(training_time)
#test_time = np.array(test_time) / np.max(test_time)
#
#plt.figure(figsize=(12, 8))
#plt.title("Score")
#plt.barh(indices, score, .2, label="score", color='r')
#plt.barh(indices + .3, training_time, .2, label="training time", color='g')
#plt.barh(indices + .6, test_time, .2, label="test time", color='b')
#plt.yticks(())
#plt.legend(loc='best')
#plt.subplots_adjust(left=.25)
#plt.subplots_adjust(top=.95)
#plt.subplots_adjust(bottom=.05)
#
#for i, c in zip(indices, clf_names):
#    plt.text(-.3, i, c)

#plt.show()