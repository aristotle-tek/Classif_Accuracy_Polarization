#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
=================================================
Classification Accuracy as a Substantive
Quantity of Interest: Measuring Polarization
in Westminster Systems

Andrew Peterson & Arthur Spirling 2017
=================================================

Generate classifier accuracy scores and predictions,
using pre-generated matrices from gen_mats.py.

"""
# Authors:
# Andrew Peterson <ajp502 at nyu dot edu>
# Arthur Spirling < at nyu dot edu>
# License: BSD 3 clause


# run with:
# python sess_estimates.py  
# or uncomment sys.argv and use python sess_estimates.py 1 '/foo/datain/' '/foo/input_mats/' '/foo/output/' 'runid'
# arguments: (1) normalize (2) input directory (3) output mat directory (4) run id

import os
import cPickle as pickle
import sys
import logging

import pandas as pd
import numpy as np
import re
import string
import itertools
import os.path
import time

import scipy
from scipy.io import mmread
from scipy.sparse import csr_matrix
from sklearn import preprocessing
from sklearn.preprocessing import maxabs_scale
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import LogisticRegression

from sklearn.cross_validation import StratifiedKFold
from sklearn.cross_validation import train_test_split


from utils import has_regex, prep_year_data, fit_pred_offline_classifiers  # Functions for preparing data, etc.

#---------------------------------------

#---------------------------------------
def randomize_by_member(df):
	""" Randomize labels by member, 
	keeping the same proportion, a la
	Gentzkow, et al 2015, 
	Measuring Polarization in High-dimensional Data"""
	#mp_ids = list(set(df.memref))
	uniq = df.drop_duplicates('memref')
	mp_ids = uniq.memref
	proportion_cons = np.mean(uniq.y_binary) # (cons = 1)
	cons = [x>(1-proportion_cons) for x in np.random.rand(1,len(mp_ids))]
	cons = [int(x) for x in cons[0] ]
	random_cons = dict(zip(mp_ids, cons))
	df['rand_cons']= df['memref'].map(random_cons)
	return(np.array(df.rand_cons))
#---------------------------------------

#---------------------------------------
def load_X_y(data_in, mats_dir, indx, yrmth, randomize, normalized=1):
	errors = []
	try:
		df = prep_year_data(data_in, yrmth, minlen=40)
		logging.info("length df: %d" % len(df))
	except:
		logging.error("failed getting year-month %s" % yrmth)
		errors.append(yrmth)
	df.index = range(len(df))
	testsize = len(df)/10

	if randomize:
		y = randomize_by_member(df)
	else:
		y = df.y_binary
	if (np.mean(y)==0 or np.mean(y)==1 or len(df)==0):
		logging.warning("no variation in year: %s" % yrmth)
		errors.append(yrmth)
		return(np.NaN, np.NaN)
	if normalized:
		X = mmread(mats_dir + 'topic_aug_mat_normalized_j5_' + str(indx) + '.mtx')
		X = csr_matrix(X)
	else:
		X = mmread(mats_dir + 'topic_aug_mat_' + str(indx) + '.mtx')
	logging.info("Num errors: %d" % len(errors))
	return(X, y)
#---------------------------------------
#
#---------------------------------------
def run_estimates(data_in, mats_dir, results_dir, run_id, sess_indx, randomize=0, normalized=1):
	""" Run classifiers for each Parliamentary session."""
	yearly_stats = {}
	yearly_preds = {}

	setup_time = []
	train_time = []
	starttime = time.time()
	errors = []
	for indx, yrmth in sess_indx.items():
		tick = time.time()
		logging.info("currently running: %s" % yrmth)
		X, y = load_X_y(data_in, mats_dir, indx, yrmth, randomize, normalized)
		st_time = time.time() - tick
		setup_time.append(st_time)
		logging.info("setup time: %d s"  % st_time)
		
		tick = time.time()
		skf = StratifiedKFold(y, n_folds=10, shuffle=True, random_state=1234)
		cls_stats = {}
		preds= {}
		foldid = 0
		for train_index, test_index in skf:
			#logging.info("fold: %d" % foldid)
			#logging.info("TRAIN: %s"  train_index)#, "TEST:", test_index)
			X_train, X_test = X[train_index], X[test_index]
			y_train, y_test = y[train_index], y[test_index]
			cls_stats[foldid], preds[foldid] = fit_pred_offline_classifiers(X_train, y_train, X_test, y_test, X)
			foldid += 1
		fit_time = time.time() - tick
		train_time.append(fit_time)
		yearly_stats[indx] = cls_stats
		yearly_preds[indx] = preds

	all_yearstime = time.time() - starttime
	logging.info("%ds required to vectorize and to fit 4 classifiers \t" % all_yearstime)
	pickle.dump(yearly_stats, open(results_dir + "/yearly_stats_rand"+ str(randomize) + "_run" + str(run_id) +".pkl", "wb" ) )
	pickle.dump(yearly_preds, open(results_dir + "/yearly_predictions_rand"+ str(randomize)+ "_run" + str(run_id)  +".pkl", "wb" ) )


#---------------------------------------
# Stats from estimates
#---------------------------------------
def avergage_per_classifier(cls_stats, classifier_names):
	accuracies = {}
	median = {}
	vs = {}
	for classif in classifier_names:
		accs = []
		for fold, stats in cls_stats.items():
			relevant = stats[classif]
			accs.append(relevant['accuracy'])
		accuracies[classif] = np.mean(accs)
		vs[classif] = np.var(accs)
		median[classif] = np.median(accs)
	return(accuracies, median, vs)
#---------------------------------------
#
#---------------------------------------
def stats_from_estimates(yearly_stats, sess_indx, randomize, run_id, results_dir):
	""" """
	classifier_names = ['SAG', 'SGD', 'Perceptron','Passive-Aggressive'] #classifiers.keys()

	rows = []
	for indx, yr in sess_indx.items():
		#logging.info(str(yr))
		try:
			curr = yearly_stats[indx]
			mns, meds, vs = avergage_per_classifier(curr, classifier_names )
			rows.append([indx, yr, mns['SAG'], mns['SGD'], mns['Perceptron'], mns['Passive-Aggressive'],
				meds['SAG'], meds['SGD'], meds['Perceptron'], meds['Passive-Aggressive'],
				vs['SAG'], vs['SGD'], vs['Perceptron'], vs['Passive-Aggressive'] ])
		except:
			logging.error("Error getting stats for: ", str(yr))

	res = pd.DataFrame(data=rows, columns = ['index', 'yrmth',
		'mn_sag','mn_sgd','mn_pcpt','mn_passAgr',
		'md_sag','md_sgd','md_pcpt','md_passAgr',
		'var_sag','var_sgd','var_pcpt','var_passAgr'	])

	res.to_csv(results_dir + '/acc_allmembers_rand' + str(randomize)+ "_run_" + str(run_id)  +".csv", index=False)
#---------------------------------------
#
#---------------------------------------
def main():
	curr_dir = os.getcwd()
	curr_dir = re.sub('/UK_data', '', curr_dir)

	logging.basicConfig(filename= curr_dir + '/log_files/sess_estimates.log',level=logging.INFO,format='%(asctime)s %(lineno)s: %(message)s')
	logging.info('Start.')
	
	#sess_indx_file = (provide path to full index for full data)
	#sess_indx = pickle.load(open(sess_indx_file, 'rb')) 
	sess_indx = {9: '1944-11', 74: '2008-12'}

	#normalize = sys.argv[1] # normalize_X = True
	randomize = 0 # sys.argv[1]
	data_in = curr_dir + "/data/" #sys.argv[2]  # data_in = '/nas/tz/uk/wtopics/'
	mats_dir = curr_dir + "/" #sys.argv[3]
	run_id = "replication" #sys.argv[4]
	results_dir = curr_dir

	run_estimates(data_in, mats_dir, results_dir, run_id, sess_indx, randomize=randomize, normalized=1)
	logging.info("Calculating accuracy stats from estimates")

	yearly_stats = pickle.load(open(results_dir + "/yearly_stats_rand"+ str(randomize) + "_run" + str(run_id) +".pkl", 'rb'))
	stats_from_estimates(yearly_stats, sess_indx, randomize, run_id, results_dir)


if __name__ == "__main__":
	main()
