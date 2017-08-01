# Replication files for "Classification Accuracy as a Substantive Quantity of Interest: Measuring Polarization in Westminster Systems"

## Overview

*Note: In these replication files, we do not provide the full (rather large) data used in the paper (see below for instructions on obtaining that data) but do provide:

- the scripts we used to generate the full results from the full XML data. These scripts have been modified trivially (changing just the session_indx dict) to work with the included sample data to generate results for two Parliamentary sessions.

- accuracy results and predicted values generated from the full data that can be used to produce the figures.*

These replication files are organized in the following way:

1. Scripts to generate simulated data and run classifiers on them.
2. UK Parliament: Scripts to convert the XML data to sparse matrices with topic dummies, then run machine learning algorithms to produce accuracy results and predicted values.
3. Scripts to generate the figures found in the article from the accuracy results and predicted values.

*NB: As dataverse does not allow for folders, the filepaths may require altering since the scripts were created with data in a subfolder called "data".*

### Prerequisites

* You will need [Python](https://www.python.org/) and additional packages ([Pandas](http://pandas.pydata.org/), [Scikit-Learn](http://scikit-learn.org/)) to convert the xml data and to generate the estimates.

* You will need [R](https://www.r-project.org/) and additional packages (ggplot2, reshape2, strucchange) to generate the figures.

* For Figure 5 you will need an [API key for the Manifesto project](https://manifestoproject.wzb.eu/information/documents/api)

### Data included

The full XML speech data was provided by [Rheault, et al (2016)](http://journals.plos.org/plosone/article?id=10.1371/journal.pone.0168843). We provide the scripts we used to convert the data as well as converted comma separated value files for selected sessions, along with the output of our machine learning classifiers as follows:

1. Speech data and covariates for sessions 1944-11 and 2008-12 (comma separated values)

2. "acc_sims.csv" (from sim_estimates.py)

3. "preds_sims.csv" (from sim_estimates.py)

4. An index of years and Parliamentary sessions: "years_session_index.csv"

5. ML classification accuracies for all sessions "acc_j27_allmembers.csv" (from sess_estimates.py)

6. "SAG_speaker_prob_estimates_allmembers_j27.csv" (from sess_estimates.py)

## Implementation

### Simulated data

1. sim_speeches.r (generates the data)
2. sim_estimates.py (run ML classifiers)

### UK Parliament: Preparing speech data, running ML classifiers

*These scripts explain our process but require the complete XML data to generate the full results. (2) and (3) can however be run on the included csv files to generate accuracy statistics and predicted values.*

1. xml2csv.py (converts xml to csv). 
Run from the command line with: `python xml2csv.py`

2. gen_mats.py (generates sparse document term matrices, augmented with topic indicators).
Run from the command line with: `python gen_mats.py`

3. sess_estimates.py (run ML classifiers) 
Run from the command line with: `python sess_estimates.py`

### Generating the figures

*NB: These scripts can be run independently as they refer to only to the included data.*

1. Figure 1 (Figure_1.r) 
2. Figure 2 (Figure_2.r)
3. Figure 3 (Figure_3.r)
4. Figure 4 (Figure_4.r)
5. Figure 5 (Figure_5.r)

### Additional Files

1. example log files in the /log_files/ directory.
2. bootstrap_notes.py (explanation of changes for generating bootstrapped estimates.)
2. xgboost_estimates.py (similar to sess_estimates.py but using the xgboost classifier)

## Authors

* [**Andrew Peterson**](https:www.andrewjerelpeterson.com) - Postdoctoral Researcher - University of Geneva

* [**Arthur Spirling**](http://www.nyu.edu/projects/spirling/) - Associate Professor of Politics and Data Science - New York University


## License

This project is open source under the BSD 3-Clause.

## Acknowledgments

* XML data was kindly shared by Kaspar Beelen. See [Rheault, L, Beelen K, Cochrane C and Hirst G. 2016. "Measuring Emotion in Parliamentary Debates with Automated Textual Analysis." PLOS ONE 11(12).](http://journals.plos.org/plosone/article?id=10.1371/journal.pone.0168843)

* Initial ML work was based on documentation from [Scikit-Learn](http://scikit-learn.org/)

