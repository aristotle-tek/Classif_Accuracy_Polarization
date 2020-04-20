#!/usr/bin/python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import re
import string
import pickle as pickle
import os
#from nltk.corpus import stopwords
import itertools

import itertools
import os.path
import time
import logging

import scipy
from scipy.sparse import csr_matrix
from sklearn import preprocessing
from sklearn.preprocessing import maxabs_scale

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split


#------------------------------------------------------------------------------------------

#------------------------------------------------------------------------------------------
def has_regex(regex, text):
	s = re.search(regex, text)
	if s:
		return(1)
	else:
		return(0)
#------------------------------------------------------------------------------------------

#------------------------------------------------------------------------------------------
def prep_year_data(pref, year, minlen=40): #, makebalanced=True):
	df = pd.read_csv(pref + 'uk-text-' + str(year) +'.csv')
	rel = [has_regex(r'(uk\.p\.Con|uk\.p\.Lab)', str(x)) for x in  df['partyref']]
	df= df[[x==1 for x in rel]]
	df['cons'] = df['party']=='Conservative'
	df['text'] = [str(x) for x in df.text]
	gt40 = [len(x)>minlen for x in df['text']]
	df = df[[x==1 for x in gt40]]
	df['y_binary'] = [int(x) for x in df['cons']]
	return(df)

#------------------------------------------------------------------------------------------

#------------------------------------------------------------------------------------------
def augment_with_topics(X, df):
	logging.info("augmenting with topic FE...")
	alltopics = list(set(df.topic))
	cols = []
	for top in alltopics[:]:
		curr = [x==top for x in df.topic]
		curr = [int(x) for x in curr]
		#print(sum(curr))
		cols.append(curr)
	topicFE = csr_matrix(cols)
	X2 = scipy.sparse.hstack((X, topicFE.transpose() ))
	X2 = csr_matrix(X2)
	return(X2, alltopics)
#------------------------------------------------------------------------------------------

#------------------------------------------------------------------------------------------
def fit_pred_offline_classifiers(X_train, y_train, X_test, y_test, X):

	classifiers_balanced = {
		'SGD': SGDClassifier(class_weight='balanced', n_jobs=10),
		'Perceptron': Perceptron(class_weight='balanced', n_jobs=10),
		'Passive-Aggressive': PassiveAggressiveClassifier(class_weight='balanced', n_jobs=10),
	}

	classifiers_bal_predprob = {"SAG": LogisticRegression(solver='sag', n_jobs=10, tol=1e-1, C=1.e4 / 50000 ),} # , C=1.e4 / 50000


	cls_stats = {}
	preds = {}
	for cls_name in classifiers_bal_predprob:
		stats = {'n_train': 0, 'n_train_pos': 0, 'accuracy': 0.0, 't0': time.time(), 'total_fit_time': 0.0}
		cls_stats[cls_name] = stats

	for cls_name in classifiers_balanced:
		stats = {'n_train': 0, 'n_train_pos': 0, 'accuracy': 0.0, 't0': time.time(), 'total_fit_time': 0.0}
		cls_stats[cls_name] = stats

	tick = time.time()

	for cls_name, cls in classifiers_bal_predprob.items():
		logging.info("fitting %s" % cls_name)
		cls.fit(X_train, y_train)#, classes=all_classes)
		preds[cls_name] = cls.predict_proba(X)
		# stats
		cls_stats[cls_name]['total_fit_time'] += time.time() - tick
		cls_stats[cls_name]['n_train'] += X_train.shape[0]
		cls_stats[cls_name]['n_train_pos'] += sum(y_train)
		cls_stats[cls_name]['accuracy'] = cls.score(X_test, y_test)

	tick = time.time()

	for cls_name, cls in classifiers_balanced.items():
		logging.info("fitting %s" % cls_name)
		cls.fit(X_train, y_train)#, classes=all_classes)
		preds[cls_name] = cls.predict(X)
		# stats
		cls_stats[cls_name]['total_fit_time'] += time.time() - tick
		cls_stats[cls_name]['n_train'] += X_train.shape[0]
		cls_stats[cls_name]['n_train_pos'] += sum(y_train)
		cls_stats[cls_name]['accuracy'] = cls.score(X_test, y_test)
	return(cls_stats, preds)
#------------------------------------------------------------------------------------------

#------------------------------------------------------------------------------------------
def speeches_per_session(sess_indx, df, datafolder, gov_only, just_backbenchers, mkbalanced=0):
	errors = []
	year_num = {}
	for indx, yrmth in sess_indx.items()[:]:
		logging.info(yrmth)
		try:
			df = prep_year_data(datafolder, yrmth, minlen=40)
			#logging.info(len(df))
		except:
			logging.error("failed getting year-month %s" % yrmth)
			errors.append(yrmth)
		if gov_only:
			df = df[df['role']=='gov']
		elif just_backbenchers:
			df = df[df['role']=='mp']
		if mkbalanced:
			df = make_balanced(df)
		df.index = range(len(df))
		year_num[yrmth] = len(df)
	return(year_num, errors)
#------------------------------------------------------------------------------------------

#------------------------------------------------------------------------------------------
