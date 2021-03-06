from __future__ import division
# from scipy import *
from matplotlib.pyplot import *
from numpy import *
from collections import Counter as C
import parser

##############
### II.1.1 ###
##############


### Load data

# train = open('IrisTrain2014.dt').read().split('\n')
# train = array( [t.split(' ') for t in train] )
# train = train.astype(float)
train = array(parser.parse('../IrisTrain2014.dt'))

# test = open('IrisTest2014.dt').read().split('\n')
# test = array( [t.split(' ') for t in test] )
# test = test.astype(float)
test = array(parser.parse('../IrisTest2014.dt'))

### Visualize data

shapes = ['^','o','s']
colors = ['b','r', 'g']

for i in train:
	scatter(i[0],i[1], c=colors[int(i[2])], marker=shapes[int(i[2])])

# show()



### calculate baseline

# from sklearn.lda import LDA
# def baselda(train, test):
# 	clf = LDA()
# 	clf.fit(train[:,:-1],train[:,-1])
# 	return clf.predict(test[:,:-1]) == test[:,-1]

# result, nresult = baselda(train, test), baselda(ntrain, ntest)

# print "Baseline for non-normalized iris: \t",result.sum()/len(result)
# print "Baseline for normalized iris: \t",nresult.sum()/len(nresult)

###########
### LDA ###
###########

# initiate variables / fitting

# classes is a set of all available classes in dataset (all k's).
classes = set(train[:,-1].astype(int))

# in slides m is the number of classes
m = len(classes)


#l_k is amount of datapoints with class k. Pr_k is Pr(Y = C_k) from slide 11 (linearClassification.pdf)
l_k, Pr_k = [], []
l = len(train)
for i in classes:
	l_k.append( sum([train[:,-1] == i ]) )
	Pr_k.append( l_k[-1]/l )


# Get the mean, mu, for each class, k
mu_k = []
for i in classes:
	mu_k.append(train[ train[:,-1] == i ][:,:-1].mean(axis = 0))

# Get predicted covariance, Sigma
Sigmas = []
for k in classes:
	# Subtract the mean of class k from all training data of class k
	val = train[train[:,-1] == k][:,:-1] - mu_k[k]
	outer_prod = array( [outer(val[j],val[j]) for j in range(len(val)) ] )
	Sigmas.append( outer_prod.sum(axis=0) )
	#Sigma.append( outer(val,val.T).sum(axis=0))
Sigma = array(Sigmas).sum(axis = 0)/ (l-m)

# /fitting


# delta_k function as defined on slide 11 (linearClassification.pdf)
def delta_k(k, test):
	global mu_k, Sigma, l, l_k, Pr_k
	Sigma_inv = linalg.inv(Sigma)
	r = []
	for x in test:
		r.append( dot( dot( x.T, Sigma_inv),  mu_k[k])  - (dot( dot(mu_k[k].T, Sigma_inv), mu_k[k] )/2) + log(Pr_k[k]) )
	r = array(r)
	return r

print delta_k(1, test[:,:-1])


##############
### II.1.2 ###
##############


# Normalize data
def normalize(data, mu, sigma):
	return (data - mu)/sigma

ntrain = copy(train)
ntest = copy(test)

mu = [train[:,i].mean() for i in [0,1]]
sigma = [train[:,i].std() for i in [0,1]]

for i in [0,1]:
	ntrain[:,i] = normalize(train[:,i],mu[i],sigma[i])
	ntest[:,i] = normalize(test[:,i],mu[i],sigma[i])




# #################
# ### Sunspots ####
# #################

# strain = open('sunspotsTrainStatML.dt').read().split('\n')
# strain = array( [t.split(' ') for t in strain] )
# strain = strain.astype(float)

# stest = open('sunspotsTestStatML.dt').read().split('\n')
# stest = array( [t.split(' ') for t in stest] )
# stest = stest.astype(float)


# ############################
# ### 1D linear regression ###
# ############################

# # See Udacity lectures on Linear Regression
# #	Intro to Artificial Intelligence > 5. Machine Learning > Linear Regression & More Linear Regression

# def train_linalg(x,y):
# 	return get_w1(x,y), get_w0(x,y)

# def get_w1(x,y):
#      M = len(x)
#      return (M*(x*y).sum()-x.sum()*y.sum())/(M*(x*x).sum()-x.sum()**2)

# def get_w0(x,y):
# 	M = len(x)
# 	w1 = get_w1(x,y)
# 	return ( y.sum() - w1*x.sum() ) / M

# def test_linalg(X,Y, w0, w1):
# 	# Note that x, y should be testing data!
# 	results = []
# 	for x, y in zip(X,Y):
# 		pred = w1*x+w0
# 		print y - pred
# 		results.append(pred)
# 	return results







# def train_linalg(x,y):
# 	return get_w(x,y), get_w0(x,y)

# def get_w1(x,y):
#      M = len(x)
#      return (M*(x*y).sum()-x.sum()*y.sum())/(M*(x*x).sum()-x.sum()**2)

# def get_w0(x,y):
# 	M = len(x)
# 	w1 = get_w1(x,y)
# 	return ( y.sum() - w1*x.sum() ) / M

# def test_linalg(X,Y, w0, w1):
# 	# Note that x, y should be testing data!
# 	results = []
# 	for x, y in zip(X,Y):
# 		pred = w1*x+w0
# 		print y - pred
# 		results.append(pred)
# 	return results
