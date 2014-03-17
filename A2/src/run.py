from __future__ import division
import scipy as sp
import matplotlib.pyplot as plot
from sklearn.lda import LDA

import lda_no_norm
import numpy as np



### Load data

train = open('../IrisTrain2014.dt').read().split('\n')
train = array( [t.split(' ') for t in train] )
train = train.astype(float)

test = open('../IrisTest2014.dt').read().split('\n')
test = array( [t.split(' ') for t in test] )
test = test.astype(float)

### calculate baseline


def baselda(train, test, norm=True):
	if norm == True:		
		clf = LDA()
	else:
		clf = lda_no_norm.LDA_no_norm()
	clf.fit(train[:,:-1],train[:,-1])
	return clf.predict(test[:,:-1]) == test[:,-1]


result = baselda(train, test, False)
print "Error for non-normalized iris, using test: \t", 1 - (result.sum()/len(result))
result = baselda(train, train, False)
print "Error for non-normalized iris, using train: \t", 1 - (result.sum()/len(result))



result = baselda(train, test)
print "Error for normalized iris, using test: \t",1 - (result.sum()/len(result))

result = baselda(train, train)
print "Error for normalized iris, using train: \t",1 - (result.sum()/len(result))