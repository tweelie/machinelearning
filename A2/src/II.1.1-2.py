from __future__ import division
import scipy as sp
import matplotlib.pyplot as plot
import numpy as np



class LDAC:
## This class is written by Davide Albanese, <albanese@fbk.eu>.
## (C) 2011 mlpy Developers.
## See http://sourceforge.net/projects/mlpy/files/
## Released under GNU General Public License
## version 3 of the License or newer
## <http://www.gnu.org/licenses/>

    """Linear Discriminant Analysis Classifier.
    """
    
    def __init__(self):
        """Initialization.
        """

        self._labels = None
        self._w = None
        self._bias = None
      
    def learn(self, x, y):
        """Learning method.

        :Parameters:
           x : 2d array_like object
              training data (N, P)
           y : 1d array_like object integer
              target values (N)
        """
        
        xarr = np.asarray(x, dtype=np.float)
        yarr = np.asarray(y, dtype=np.int)
        
        if xarr.ndim != 2:
            raise ValueError("x must be a 2d array_like object")
        
        if yarr.ndim != 1:
            raise ValueError("y must be an 1d array_like object")

        self._labels = np.unique(yarr)
        k = self._labels.shape[0]

        if k < 2:
            raise ValueError("number of classes must be >= 2")     
        
        p = np.empty(k, dtype=np.float)
        mu = np.empty((k, xarr.shape[1]), dtype=np.float)
        cov = np.zeros((xarr.shape[1], xarr.shape[1]), dtype=np.float)

        for i in range(k):
            wi = (yarr == self._labels[i])
            p[i] = np.sum(wi) / float(xarr.shape[0])
            mu[i] = np.mean(xarr[wi], axis=0)
            xi = xarr[wi] - mu[i]
            cov += np.dot(xi.T, xi)
        cov /= float(xarr.shape[0] - k)
        covinv = np.linalg.inv(cov)
        
        self._w = np.empty((k, xarr.shape[1]), dtype=np.float)
        self._bias = np.empty(k, dtype=np.float)

        for i in range(k):           
            self._w[i] = np.dot(covinv, mu[i])
            self._bias[i] = - 0.5 * np.dot(mu[i], self._w[i]) + \
                np.log(p[i])

    def labels(self):
        """Outputs the name of labels.
        """
        
        return self._labels
        
    def w(self):
        """Returns the coefficients.
        For multiclass classification this method returns a 2d 
        numpy array where w[i] contains the coefficients of label i.
        For binary classification an 1d numpy array (w_1 - w_0) 
        is returned.
        """
        
        if self._w is None:
            raise ValueError("no model computed.")

        if self._labels.shape[0] == 2:
            return self._w[1] - self._w[0]
        else:
            return self._w

    def bias(self):
        """Returns the bias.
        For multiclass classification this method returns a 1d 
        numpy array where b[i] contains the coefficients of label i. 
        For binary classification an float (b_1 - b_0) is returned.
        """
        
        if self._w is None:
            raise ValueError("no model computed.")
        
        if self._labels.shape[0] == 2:
            return self._bias[1] - self._bias[0]
        else:
            return self._bias

    def pred(self, t):
        """Does classification on test vector(s) `t`.
      
        :Parameters:
            t : 1d (one sample) or 2d array_like object
                test data ([M,] P)
            
        :Returns:        
            p : integer or 1d numpy array
                predicted class(es)
        """
        
        if self._w is None:
            raise ValueError("no model computed.")

        tarr = np.asarray(t, dtype=np.float)

        if tarr.ndim == 1:
            delta = np.empty(self._labels.shape[0], dtype=np.float)
            for i in range(self._labels.shape[0]):
                delta[i] = np.dot(tarr, self._w[i]) + self._bias[i]
            return self._labels[np.argmax(delta)]
        else:
            delta = np.empty((tarr.shape[0], self._labels.shape[0]),
                        dtype=np.float)
            for i in range(self._labels.shape[0]):
                delta[:, i] = np.dot(tarr, self._w[i]) + self._bias[i]
            return self._labels[np.argmax(delta, axis=1)]


def normalize(ndata, ntest):
	# Normalizes test and training data using mean and std from training data
	mu = ndata.mean(axis=0)
	sigma = ndata.std(axis=0)
	return (ndata - mu) / sigma, (ntest - mu) / sigma 

def run_lda(train, train_t, test):
	lda = LDAC()
	lda.learn(train, train_t)
	pred = lda.pred(test)

	return pred
	

### Load data

train = open('../IrisTrain2014.dt').read().strip().split('\n')
train = np.array( [t.split(' ') for t in train] )
train_t = train[:,-1].astype(int)
train = train[:,:-1].astype(float)

test = open('../IrisTest2014.dt').read().strip().split('\n')
test = np.array( [t.split(' ') for t in test] )
test_t = test[:,-1].astype(int)
test = test[:,:-1].astype(float)

ntrain, ntest = normalize(train, test)


###############################
### print result for II.1.1 ###
###############################

pred = run_lda(train, train_t, test)
print "Error for non-normalized iris, using test: \t", 1-(pred == test_t).mean()
pred = run_lda(train, train_t, train)
print "Error for non-normalized iris, using train: \t", 1-(pred == train_t).mean()

###############################
### print result for II.1.2 ###
###############################

pred = run_lda(ntrain, train_t, ntest)
print "Error for normalized iris, using test: \t", 1-(pred == test_t).mean()
pred = run_lda(ntrain, train_t, ntrain)
print "Error for normalized iris, using train: \t", 1-(pred == train_t).mean()
