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

run with "python xgboost_estimates.py"


To run this you will need to download XGBoost. See:
http://xgboost.readthedocs.io/en/latest/python/python_intro.html
"""

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


from sklearn.cross_validation import StratifiedKFold
from sklearn.cross_validation import train_test_split

import xgboost as xgb


#---------------------------------------

#---------------------------------------
def has_regex(regex, text):
	s = re.search(regex, text)
	if s:
		return(1)
	else:
		return(0)
#---------------------------------------

#---------------------------------------
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
#---------------------------------------

#---------------------------------------
def fit_pred_xgb(X_train, y_train, X_test, y_test, X, predict=False):
	opt_depth = 14
	opt_estim = 400
	cls = xgb.XGBClassifier(max_depth=opt_depth, n_estimators= opt_estim, seed=42, nthread=-1)#, gamma=1)

	cls_stats = {}
	preds = {}
	cls_name = 'xgb'
	stats = {'n_train': 0, 'n_train_pos': 0, 'accuracy': 0.0, 't0': time.time(), 'total_fit_time': 0.0}
	cls_stats[cls_name] = stats
	tick = time.time()

	print("fitting %s" % cls_name)
	cls.fit(X_train, y_train)#, classes=all_classes)
	if predict:
		preds[cls_name] = cls.predict_proba(X)
	else:
		preds[cls_name] = np.NaN
	# stats
	cls_stats[cls_name]['total_fit_time'] += time.time() - tick
	cls_stats[cls_name]['n_train'] += X_train.shape[0]
	cls_stats[cls_name]['n_train_pos'] += sum(y_train)
	cls_stats[cls_name]['accuracy'] = cls.score(X_test, y_test)
	print("Accuracy: ", cls.score(X_test, y_test))

	tick = time.time()

	return(cls_stats, preds)
#------------------------------------------------------------------------------------------

#------------------------------------------------------------------------------------------
def randomize_by_member(df):
	""" Randomize labels by member, 
	keeping the same proportion, a la Gentzkow, et al 2015, 
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

	predict = False # NB: xgboost has trouble with sparse matrices, to predict you may need to use .toarray()
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
			logging.info("fold: %d" % foldid)
			#logging.info("TRAIN: %s"  train_index)#, "TEST:", test_index)
			X_train, X_test = X[train_index], X[test_index]
			y_train, y_test = y[train_index], y[test_index]
			cls_stats[foldid], preds[foldid] = fit_pred_xgb(X_train, y_train, X_test, y_test, X, predict=False)
			foldid += 1
		fit_time = time.time() - tick
		train_time.append(fit_time)
		yearly_stats[indx] = cls_stats
		yearly_preds[indx] = preds

	all_yearstime = time.time() - starttime
	logging.info("%ds required to vectorize and to fit classifier \t" % all_yearstime)
	pickle.dump(yearly_stats, open(results_dir + "/yearly_stats_rand"+ str(randomize) + "_run" + str(run_id) +".pkl", "wb" ) )
	if predict:
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
	classifier_names = ['xgb'] #classifiers.keys()

	rows = []
	for indx, yr in sess_indx.items():
		logging.info(str(yr))
		try:
			curr = yearly_stats[indx]
			mns, meds, vs = avergage_per_classifier(curr, classifier_names )
			rows.append([indx, yr, mns['xgb'], vs['xgb'] ])
		except:
			logging.error("Error getting stats for: ", str(yr))

	res = pd.DataFrame(data=rows, columns = ['index','yrmth',
		'mn_xgb', 'var_xgb'])

	res.to_csv(results_dir + '/acc_allmembers_rand' + str(randomize)+ "_run_" + str(run_id)  +".csv", index=False)
#---------------------------------------
#
#---------------------------------------
def main():
	curr_dir = os.getcwd()
	curr_dir = re.sub('/additional_files', '', curr_dir)

	logging.basicConfig(filename= curr_dir + '/log_files/xgb_estimates.log',level=logging.INFO,format='%(asctime)s %(lineno)s: %(message)s')
	logging.info('Start.')
	
	#sess_indx_file = (provide path to full index for full data)
	#sess_indx = pickle.load(open(sess_indx_file, 'rb')) 
	sess_indx = {9: '1944-11', 74: '2008-12'}

	#normalize = sys.argv[1] # normalize_X = True
	randomize = 0 # sys.argv[1]
	data_in = curr_dir + "/data/" #sys.argv[2]  # data_in = '/nas/tz/uk/wtopics/'
	mats_dir = curr_dir + "/" #sys.argv[3]
	run_id = "xgb_replic" #sys.argv[4]
	results_dir = curr_dir

	run_estimates(data_in, mats_dir, results_dir, run_id, sess_indx, randomize=randomize, normalized=1)
	logging.info("Calculating accuracy stats from estimates")

	yearly_stats = pickle.load(open(results_dir + "/yearly_stats_rand"+ str(randomize) + "_run" + str(run_id) +".pkl", 'rb'))
	stats_from_estimates(yearly_stats, sess_indx, randomize, run_id, results_dir)


if __name__ == "__main__":
	main()
