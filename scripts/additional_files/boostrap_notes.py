""" A note on bootstrapping estimates:
To generate the bootstrap estimates we modify the 
sess_estimates.py script only slightly to include an additional loop
in the run_estimates(.) function, then gather these additional results.
"""

from sklearn.utils import resample

#[...]

for bootstrapi in range(num_bootstraps):
	X_index = range(X.shape[0])
	resamp = resample(X_index, random_state=9889)

	ycurr = y[resamp]
	Xcurr = X[resamp]

	ycurr.index = range(len(ycurr))

	skf = StratifiedKFold(y, n_folds=10, shuffle=True, random_state=1234)
	#[...]