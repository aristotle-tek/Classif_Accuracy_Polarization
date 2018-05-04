#!/usr/bin/env python

"""
=================================================
Classification Accuracy as a Substantive
Quantity of Interest: Measuring Polarization
in Westminster Systems

Andrew Peterson & Arthur Spirling 2017
=================================================

Parse XML data for relevant data, 
output to csv files and generate session_index.pkl.

Inputs: years_d4.csv.


---speeches are in:
root.proceedings.topic.scene.speech
--- other info is in:
root.meta
root.proceedings.topic.stage-direction

"""


from datetime import datetime,date,time
import sys
import pandas as pd
import numpy as np
import os
import re
from lxml import objectify
from lxml import etree
from glob import glob
import fnmatch
import string
import cPickle as pickle
import codecs
import functools

import roman
import logging




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
def remove_bracket_html_inkeys(dictin):
	keys = dictin.keys()
	keys = [re.sub(r'{http:\/\/www\.politicalmashup\.nl}', '', x) for x in keys]
	newdict = dict(zip(keys, dictin.values()))
	return(newdict)
#------------------------------------------------------------------------------------------

#------------------------------------------------------------------------------------------
def get_attributes(att):
	try:
		idn = att['id']
	except:
		idn = ''
	try:
		speaker = att['speaker']
	except:
		speaker = ''
	try:
		role = att['role']
	except:
		role = ''
	try:
		party = att['party']
	except:
		party = ''
	try:
		partyref = att['party-ref']
	except:
		partyref = ''
	try:
		memref = att['member-ref']
	except:
		memref = ''
	return(idn, speaker, role, party, partyref, memref)
#------------------------------------------------------------------------------------------

#------------------------------------------------------------------------------------------
def get_all_text(elem):
	alltext = ''
	childs = elem.getchildren()
	for ch in childs:
		isp = has_regex(r'p$', ch.tag)
		if isp:
			if ch.text is not None:
				alltext = alltext + ch.text + ' '
	return(alltext.strip())
#------------------------------------------------------------------------------------------

#------------------------------------------------------------------------------------------
def update_topic(elem, curr_topic):
	try:
		att = elem.attrib
		att = remove_bracket_html_inkeys(att)
		newtopic = att['title'] # could instead use 'id'
		return(newtopic)
	except:
		print('error, using existing topic')
		logging.error('error getting topic')
		#logging.error('error getting topic: %s' % att)
		return(curr_topic)
#------------------------------------------------------------------------------------------

#------------------------------------------------------------------------------------------
""" Recursively get the topic and text of a speech and append as a row to a list.
NB: only pass curr_topic down the tree, not to next branch. """

def recursive_get_speeches_topics(element, rows, curr_topic, filedate):
	for elem in element.getchildren():
		isspeech = has_regex(r'speech', elem.tag)
		istopic = has_regex(r'topic', elem.tag)
		if istopic:
			curr_topic = update_topic(elem, curr_topic)
		if isspeech:
			try:
				att = elem.attrib
				att = remove_bracket_html_inkeys(att)
				idn, speaker, role, party, partyref, memref = get_attributes(att)
				text = get_all_text(elem)
				rows.append([idn, curr_topic, role, speaker, party, partyref, memref, filedate, text]) # encode('utf-8')
			except:
				print("failed getting speech: %s" % elem.attrib)
				logging.error("failed getting speech")
		else:
			rows = recursive_get_speeches_topics(elem, rows, curr_topic, filedate)
	return(rows)
#------------------------------------------------------------------------------------------

#------------------------------------------------------------------------------------------
def recursive_get_speeches(element, rows, filedate):
	for elem in element.getchildren():
		isspeech = has_regex(r'speech', elem.tag)
		if isspeech:
			try:
				#print sp.tag
				att = elem.attrib
				att = remove_bracket_html_inkeys(att)
				idn, speaker, role, party, partyref, memref = get_attributes(att)
				text = get_all_text(elem)
				rows.append([idn, role, speaker, party, partyref, memref, filedate, text]) # encode('utf-8')
			except:
				print("failed getting speech: %s" % elem.attrib)
				logging.error("failed getting speech")
		else:
			recursive_get_speeches(elem, rows, filedate)
	return(rows)
#------------------------------------------------------------------------------------------

#------------------------------------------------------------------------------------------
def date_from_filename(filename):
	s = re.search(r'd\.([0-9]{4}\-[0-9]{2}\-[0-9]{2})', filename)
	if s:
		date = pd.to_datetime(s.group(1))
		return(date)
	else:
		print("error getting date")
		logging.error('failed getting date.')
		return(np.NaN)
#------------------------------------------------------------------------------------------

#------------------------------------------------------------------------------------------
def remove_bracketnums(text):
	textout = re.sub(r'\[[0-9]{1,10}\]', '', text)
	textout = re.sub(r'\[R\]', '', textout)
#	textout = re.sub(r'\[[0-9A-Za-z]{5,10}\]', '', textout)
	return(textout)
#------------------------------------------------------------------------------------------

#------------------------------------------------------------------------------------------
def parse_xml_between_dates(startdate, enddate, data_in_path, output_folder):
	rows = []
	for filename in glob(data_in_path + '*'):
		logging.debug(filename)
		filedate = date_from_filename(filename)
		within1 = filedate>=startdate
		within2 = filedate<=enddate
		if within1 and within2:
			try:
				parsed = objectify.parse(open(filename))
				root = parsed.getroot()
			except:
				print("failed parsing xml for file %s" % filename)
				logging.error("failed parsing xml for file %s" % filename)
			rows = recursive_get_speeches_topics(root, rows, '-', filedate)
			print("# rows: %d" % len(rows))
	df = pd.DataFrame(data=rows, columns = ['id','topic', 'role','speaker','party','partyref','memref','filedate','text'])
	savefile = output_folder + 'uk-text-' + str(startdate.year) + '-' + str(startdate.month) + '.csv'
	print(savefile)
	df['text'] = [remove_bracketnums(x) for x in df['text']]
	df.to_csv(savefile, index=False, encoding='utf-8')
#------------------------------------------------------------------------------------------

#------------------------------------------------------------------------------------------
def main():
	curr_dir = os.getcwd()

	logging.basicConfig(filename=curr_dir + "xml2csv.log", level=logging.INFO)
	logging.info('Start.')

	data_in_path = '/ssd/andrew/data/uk/uk-proc-updated-10-05/'

	output_folder = '/nas/tz/uk/wtopics/'

	dt = pd.read_csv('/ssd/andrew/data/uk/gen/years_d4.csv')

	dt['date'] = pd.to_datetime(dt['date'])
	dates = dt['date']

	d2 = dates[1:] # (to get enddates)
	d2 = d2.append(pd.DataFrame(data=[pd.to_datetime('2015-05-01')]))
	d2.index = range(len(d2))
	dt['enddate'] = d2-1 # end 1 day before.
	dt = dt[:-1]



	ind = {}
	for i in dt.index[:]:
		startdate = dt['date'][i]
		enddate = dt['enddate'][i]
		ind[i] = str(startdate.year) + '-' + str(startdate.month)
		print("Startdate: %s" % startdate)
		logging.info("Startdate: %s" % startdate)
		parse_xml_between_dates(startdate, enddate, data_in_path, output_folder)

	pickle.dump(ind, open("/ssd/andrew/data/uk/gen/" + "session_index.pkl", "wb" ) )


if __name__ == "__main__":
	main()




#------------------------------------------------------------------------------------------
#   end.
#------------------------------------------------------------------------------------------
