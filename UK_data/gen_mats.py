#!/usr/bin/env python
""" NB: This has been modified from it's original to 
work with just the two sessions of data provided. However,
if all data is available one need only change the
 "sess_indx" dict to include all sessions.


=================================================
Classification Accuracy as a Substantive
Quantity of Interest: Measuring Polarization
in Westminster Systems

Andrew Peterson & Arthur Spirling 2017
=================================================


Generate term-document matrices
for UK Parliamentary speeches, 
augmented with topic-indicators

Inputs:
	(1) csv data (from xml2csv.py script)
	(2) session_index.pkl (from xml2csv.py script) 

Outputs:
	(1) vocab_min200.pkl
	(2) sparse scipy matrices and session_topics.pkl.

"""


# Authors:
# Andrew Peterson, NYU <andrew.peterson at unige dot ch>
# Arthur Spirling, NYU
# License: BSD 3 clause


# run with:
# python gen_mats.py

import os
import cPickle as pickle
import sys
import logging

import pandas as pd
import numpy as np
import re
import string

from glob import glob
import itertools
import os.path
import time

import scipy
from scipy.io import mmwrite
from scipy.sparse import csr_matrix
from sklearn import preprocessing
from sklearn.preprocessing import maxabs_scale
from sklearn.feature_extraction.text import CountVectorizer

from utils import has_regex, prep_year_data, augment_with_topics

#----------------------------------------------------------
# Identify vocab & save
#----------------------------------------------------------
def gen_fixed_vocab(data_in, mat_dir, sess_indx):
	print("generating vocab...")
	dfall = pd.DataFrame()
	for indx, yrmth in sess_indx.items()[:79]:
		print(yrmth)
		df = prep_year_data(data_in, yrmth, minlen=40)
		dfall = pd.concat([dfall, df])
		#logging.info(dfall.info())

	vectorizer = CountVectorizer(decode_error='ignore', min_df =200)
	text = dfall.text
	dfall = []
	X = vectorizer.fit_transform(text)
	wordlist = vectorizer.get_feature_names()
	wordlist2 = [x for x in wordlist if not has_regex(r'[0-9]', x) ]
	pickle.dump(wordlist2, open(mat_dir + "vocab_min200.pkl", "wb"))
	logging.info("Length vocab: %d" % len(wordlist))
	logging.info(str(len(text)))
	worddict = vectorizer.vocabulary_
	pickle.dump(worddict, open(mat_dir + "vocab_min200_freqdict.pkl", "wb" ) )
	wd = pd.DataFrame(data=zip(worddict.keys(), worddict.values()), columns=['word','count'])
	wd.to_csv(mat_dir + "vocab_min200_freqDF.csv", index=False, encoding='utf-8')


#--------------------------------------------------
# generate the matrices
#--------------------------------------------------
def gen_mats(data_in, mat_dir, sess_indx, normalize):
	print("generating matrices...")
	sess_topics = {}
	vocab = pickle.load(open(mat_dir + "vocab_min200.pkl", 'rb'))
	vectorizer = CountVectorizer(decode_error='ignore', vocabulary=vocab)
	errors = []

	for indx, yrmth in sess_indx.items()[:79]:
		logging.info("Starting session: %s " % yrmth)
		#df = pd.DataFrame()
		try:
			logging.info("loading data...")
			df = prep_year_data(data_in, yrmth, minlen=40)
			logging.info(str(len(df)))
		except:
			logging.error("failed getting year-month %s" % yrmth)
			errors.append(yrmth)
		df.reset_index(inplace=True)

		y = df.y_binary
		if (np.mean(y)==0 or np.mean(y)==1 or len(df)==0):
			logging.warning("no variation in year: ", yrmth)
			errors.append(yrmth)
			continue
		logging.info("vectorizing text...")
		X = vectorizer.fit_transform(df.text)

		logging.info("vocab shape: %s" % str(X.shape))

		#if aug_w_topics:
		X, topics = augment_with_topics(X, df)
		sess_topics[indx] = topics
		logging.info("writing data...")
		if normalize:
			X = maxabs_scale(X) # not sparse: X = preprocessing.scale(X)
			logging.info("w/ topics: %s" % str(X.shape))
			mmwrite(mat_dir + 'topic_aug_mat_normalized_j5_' + str(indx) + '.mtx' , X)
			logging.info("saved augmented,normalized matrix")
			logging.info("normalized X.")
		else:
			logging.info("w/ topics: %s" % str(X.shape))
			mmwrite(mat_dir + 'topic_aug_mat_' + str(indx) + '.mtx' , X)
			logging.info("saved augmented matrix")
	logging.info("Done generating matrices.")
	logging.error("errors: %d" % len(errors))

	pickle.dump(sess_topics, open(mat_dir + "session_topics.pkl", "wb" ) )
#--------------------------------------------------

#--------------------------------------------------


def main():
	

	curr_dir = os.getcwd()
	curr_dir = re.sub('/UK_data', '', curr_dir)

	logging.basicConfig(filename= curr_dir + '/log_files/gen_mats.log',level=logging.INFO,format='%(asctime)s %(lineno)s: %(message)s')
	logging.info('Start.')
	
	normalize = 1 # sys.argv[1]
	data_in = curr_dir + "/data/" #sys.argv[2]
	mat_dir = curr_dir + "/" #sys.argv[3]
	#sess_indx_file = sys.argv[4] # load sess_indx_file for full data.
	#sess_indx = pickle.load(open(sess_indx_file, 'rb')) 
	sess_indx = {9: '1944-11', 74: '2008-12'}
	gen_fixed_vocab(data_in, mat_dir, sess_indx)

	gen_mats(data_in, mat_dir, sess_indx, normalize)


if __name__ == "__main__":
	main()
