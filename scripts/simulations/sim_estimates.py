# generates accuracy and predictions using ML classifiers
# Requires generation of simulated data from "sim_speeches.r"

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
from scipy.sparse import csr_matrix
from sklearn import preprocessing
from sklearn.preprocessing import maxabs_scale
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import LogisticRegression

from sklearn.cross_validation import StratifiedKFold


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
		print("fitting %s" % cls_name)
		#logging.info("fitting %s" % cls_name)
		cls.fit(X_train, y_train)#, classes=all_classes)
		preds[cls_name] = cls.predict_proba(X)
		# stats
		cls_stats[cls_name]['total_fit_time'] += time.time() - tick
		cls_stats[cls_name]['n_train'] += X_train.shape[0]
		cls_stats[cls_name]['n_train_pos'] += sum(y_train)
		cls_stats[cls_name]['accuracy'] = cls.score(X_test, y_test)

	tick = time.time()

	for cls_name, cls in classifiers_balanced.items():
		#logging.info("fitting %s" % cls_name)
		cls =  LogisticRegression(solver='sag', n_jobs=10, tol=1e-1, C=1.e4 / X_train.shape[0]) # put this here to get C correct
		cls.fit(X_train, y_train)#, classes=all_classes)
		preds[cls_name] = cls.predict(X)
		# stats
		cls_stats[cls_name]['total_fit_time'] += time.time() - tick
		cls_stats[cls_name]['n_train'] += X_train.shape[0]
		cls_stats[cls_name]['n_train_pos'] += sum(y_train)
		cls_stats[cls_name]['accuracy'] = cls.score(X_test, y_test)
	return(cls_stats, preds)
#---------------------------------------

#---------------------------------------
def run_estimates(X, y):
#	tick = time.time()
	skf = StratifiedKFold(y, n_folds=10, shuffle=True)#, random_state=1234)
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
	#for _, stats in sorted(cls_stats.items()):
	#	accuracy, n_examples = zip(*stats['accuracy_history'])
#	fit_time = time.time() - tick
	return(cls_stats, preds)
#---------------------------------------

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
def stats_from_estimates(yearly_stats, randomize, run_id):
	""" """
	classifier_names = ['SAG', 'SGD', 'Perceptron','Passive-Aggressive'] #classifiers.keys()

	rows = []
	for indx, yr in sess_indx.items()[:79]:
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

	res.to_csv(results_dir + '/acc_allmembers_rand' + str(randomize)+ "_run" + str(run_id)  +".csv", index=False)
#---------------------------------------
#
#---------------------------------------
classifier_names = ['SAG', 'SGD', 'Perceptron','Passive-Aggressive']



def calc_xfold_acc(df, rows):
	X = np.array(df.iloc[:,1:])
	y = np.array(df.Y)
	stats, preds = run_estimates(X, y)
	mns, meds, vs = avergage_per_classifier(stats, classifier_names )

	rows.append([strength, size, separation, noisefrac, mns['SAG'], mns['SGD'], mns['Perceptron'], mns['Passive-Aggressive'],
				meds['SAG'], meds['SGD'], meds['Perceptron'], meds['Passive-Aggressive'],
				vs['SAG'], vs['SGD'], vs['Perceptron'], vs['Passive-Aggressive'] ])
	return(rows)
#---------------------------------------
#
#---------------------------------------
def calc_xfold_acc_predprob(df, rows):
	""" and predict accuracy."""
	X = np.array(df.iloc[:,1:])
	y = np.array(df.Y)
	stats, preds = run_estimates(X, y)
	folds_preds = []
	for fold in range(0,10):
		pred = preds[fold]
		probs = pred['SAG'][:,1]
		folds_preds.append(probs)
	folds_preds = np.array(folds_preds)
	folds_preds = folds_preds.mean(axis=0)

	mns, meds, vs = avergage_per_classifier(stats, classifier_names )

	rows.append([strength, size, separation, noisefrac, mns['SAG'], mns['SGD'], mns['Perceptron'], mns['Passive-Aggressive'],
				meds['SAG'], meds['SGD'], meds['Perceptron'], meds['Passive-Aggressive'],
				vs['SAG'], vs['SGD'], vs['Perceptron'], vs['Passive-Aggressive'], folds_preds])
	return(rows)
#---------------------------------------
#
#---------------------------------------


def main():
	curr_dir = os.getcwd()

	strength = 300
	size= 100

	# Accuracy, for Figure 1:
	rows = []

	for separation in np.arange(0.3, 0.5, .01):
		print("separation: %.2f" % separation)
		for noisefrac in np.arange(0,.95, .05):
			print("noise frac: %.2f" % noisefrac)
			filein = "/nas/tz/uk/sim/sims/sim_" + str(strength) + "_" + str(size) + "_" + str(int(separation*100)) + "_" + str(int(noisefrac*100)) + "_" + ".csv"
			print(filein)
			df = pd.read_csv(filein, index_col=0)
			rows = calc_xfold_acc(df, rows)


	res2 = pd.DataFrame(data=rows, columns = ['strength', 'size', 'separation','noisefrac',
		'mn_sag','mn_sgd','mn_pcpt','mn_passAgr',
		'md_sag','md_sgd','md_pcpt','md_passAgr',
		'var_sag','var_sgd','var_pcpt','var_passAgr'	])

	res2.to_csv(curr_dir + "acc_sims.csv")


	# now, predictions, with just noisefrac variation (for Figure 2)
	separation = 0.4

	rows = []

	for noisefrac in np.arange(0,1, .001):
		print("noise frac: %.2f" % noisefrac)
		filein = "/nas/tz/uk/sim/sims/sim_1k__" + str(strength) + "_" + str(size) + "_" + str(int(separation*100)) + "_" + str(noisefrac*100) + "_" + ".csv"
		filein2 = re.sub(r'.0_.csv', r'_.csv', filein)
		print(filein2)
		df = pd.read_csv(filein2, index_col=0)
		rows = calc_xfold_acc_predprob(df, rows)

	resP = pd.DataFrame(data=rows, columns = ['strength', 'size', 'separation','noisefrac',
		'mn_sag','mn_sgd','mn_pcpt','mn_passAgr',
		'md_sag','md_sgd','md_pcpt','md_passAgr',
		'var_sag','var_sgd','var_pcpt','var_passAgr', 'preds'	])

	#resP.to_csv() - not used...

	preds = resP['preds']
	p2 = []
	for i in range(1000):
		p2.append(preds[i])
	p2 = np.array(p2)

	df  = pd.DataFrame(p2)
	left =  ['l' + str(x) for x in range(1,301)]
	right =  ['r' + str(x) for x in range(1,301)]
	df.columns = left  + right
	df.to_csv(curr_dir + "preds_sims.csv", index= 0)
	print("done.")


if __name__ == "__main__":
	main()



