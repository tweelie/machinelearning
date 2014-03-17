from __future__ import division
from scipy import *
from matplotlib.pyplot import *
from numpy import *


##############
### II.2.1 ###
##############


strain = open('../sunspotsTrainStatML.dt').read().strip().split('\n')
strain = array( [t.split(' ') for t in strain] )
strain = strain.astype(float)
stargets = strain[:,-1]


stest = open('../sunspotsTestStatML.dt').read().strip().split('\n')
stest = array( [t.split(' ') for t in stest] )
stest = stest.astype(float)


def phi(x, j):
	return x[j]

def RMS(targets, pred):
	return sqrt( ((targets - pred) ** 2).mean() )


def build_PHI(X):
	out = []
	for x in X:
		out.append([phi(x,j) for j in range(len(x))])
	return array(out)


def get_weights(PHI, t):
	return dot(linalg.pinv(PHI), t)

def predict(w, test):
	l = len(test)
	return array([w[i] * test[i] for i in range(l)] ).sum()

def test(w, test, target):
	pred = array([predict(w, x) for x in test])
	return RMS(target, pred)

def pad(a):
	# adds a column of ones to a np.array
	return hstack(( ones((a.shape[0],1)), a ))

train_t = strain[:,-1]
test_t = stest[:,-1]


# Selection 1
train_sel1 = pad(strain[:,2:4])
test_sel1 = pad(stest[:,2:4])

W = get_weights(build_PHI(train_sel1), train_t)
print "RMS result from selection 1", test(W, test_sel1, test_t)

# Selection 2
train_sel2 = pad(strain)[:,[0,5]]
test_sel2 = pad(stest)[:,[0,5]]

W = get_weights(build_PHI(train_sel2), train_t)
predic_test2 = array([predict(W, x) for x in test_sel2])
predict_train2 = array([predict(W, x) for x in train_sel2])
print "RMS result from selection 2", test(W, test_sel2, test_t)


# Selection 3
train_sel3 = pad(strain[:,:-1])
test_sel3 = pad(stest[:,:-1])

W = get_weights(build_PHI(train_sel3), train_t)
print "RMS result from selection 3",test(W, test_sel3, test_t)



# real observations (target values)
r_t = hstack((strain[:,-1],stest[:,-1]))

# predictions
W = get_weights(build_PHI(train_sel1), train_t)
r_1 = array([predict(W, x) for x in vstack((train_sel1,test_sel1)) ])
W = get_weights(build_PHI(train_sel2), train_t)
r_2 = array([predict(W, x) for x in vstack((train_sel2,test_sel2)) ])
W = get_weights(build_PHI(train_sel3), train_t)
r_3 = array([predict(W, x) for x in vstack((train_sel3,test_sel3)) ])


#All observed data for selection 2
x_2 = hstack((train_sel2[:,-1],test_sel2[:,-1]))


### x vs t for selection 2
figure(1)
plot(x_2,r_t, 'bs', label="Observed Sunspots, vs data")
plot(x_2, r_2, 'rs', label="Predicted Sunspots, vs data")
title("Sunspot prediction vs observations. Selection 2")
legend(loc=2)
### sunspot vs. predictions


figure(2)

years = array(range(len(r_1)))+1715
suptitle("Observed and predicted sunspots over time")
for i, r in enumerate((r_1, r_2, r_3)):
	subplot(1,3,i+1)
	plot(years,r_t, 'b', label="Observed sunspots")
	plot(years,r, 'r', label="Predicted sunspots")
	title("Selection " +str(i+1))
	legend(loc=2)
show()
















