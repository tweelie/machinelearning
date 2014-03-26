#! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import math
import svmutil as svm

import string

def parse(filename):
    xs = []
    ts = []
    for line in open(filename).readlines():
        cols = string.split(line)
        xs += [map(float, cols[:-1])]
        ts += [int(cols[-1])]
    return (xs, map(lambda t: (t*2)-1, ts)) # ts mapped from [0, 1] -> [-1, 1]

def mean_and_var(list):
    mean = sum(list)/len(list)
    vec_zero_mean = np.matrix(list).T - mean
    var = (vec_zero_mean.T * vec_zero_mean)/len(list)
    return (mean, var.A1[0])

def normalizer(list):
    (mean, var) = mean_and_var(list)
    return lambda new_list: map(lambda x: (x-mean)/math.sqrt(var), new_list)

def apply(fun, list):
    return fun(list)

def many_normalizer(mat):
    trans_dlist = mat.T.A
    return lambda new_mat: np.matrix(map(apply, map(normalizer, trans_dlist), new_mat.T.A)).T

def partition(folds):
    return lambda li: [([li[j] for j in range(len(li)) if j%folds != i], li[i::folds]) for i in range(folds)]

# test_li = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
# print(str(map(partition(5), (test_li, test_li[::-1]))))

def raw_eval((train_xs, test_xs), (train_ts, test_ts), c, gamma):
    print("\nTraining without normalization, C: "+str(c)+", gamma: "+str(gamma))
    param = svm.svm_parameter('-s '+str(svm.C_SVC)+' -t '+str(svm.RBF)+' -g '+str(gamma)+' -c '+str(c))
    prob = svm.svm_problem(train_ts, train_xs)
    model = svm.svm_train(prob, param)
    return svm.svm_predict(test_ts, test_xs, model)

def norm_eval((train_xs, test_xs), (train_ts, test_ts), c, gamma):
    print("\nTraining with normalization, C: "+str(c)+", gamma: "+str(gamma))
    fun_norm = many_normalizer(np.matrix(train_xs))
    norm_train_xs = fun_norm(np.matrix(train_xs)).A.tolist()
    norm_test_xs = fun_norm(np.matrix(test_xs)).A.tolist()
    param = svm.svm_parameter('-s '+str(svm.C_SVC)+' -t '+str(svm.RBF)+' -g '+str(gamma)+' -c '+str(c))
    prob = svm.svm_problem(train_ts, norm_train_xs)
    model = svm.svm_train(prob, param)
    return svm.svm_predict(test_ts, norm_test_xs, model)

def accumulator((t_acc, t_ms_err, t_n), (pts, (acc, ms_err, corr), p_ests)):
    return (t_acc+(acc*len(pts)), t_ms_err + (ms_err*len(pts)), t_n+len(pts))

def cross_validate(data, folds, c, gamma, fun_eval):
    (part_xs, part_ts) = tuple(map(partition(folds), data))
    res = map(lambda xs, ys: fun_eval(xs, ys, c, gamma), part_xs, part_ts)
    (t_acc, t_ms_err, t_n) = reduce(accumulator, res, (0, 0, 0))
    return ((c, gamma), (t_acc/t_n, t_ms_err/t_n))

def grid_search((train_xs, train_ts), (test_xs, test_ts), fun_eval):
    res = map(lambda (c, gamma):
              cross_validate((train_xs, train_ts), 5, c, gamma, fun_eval),
                [(pow(10, x), pow(10, y)) for x in range(-2, 5)
                    for y in range(-4, 3)])
    print("\n")
    for ((c, gamma), (acc, ms_err)) in res:
        print("C: "+str(c)+"\ngamma: "+str(gamma)+"\nAccuracy: "+str(acc)+"%\nRMS Error: "+str(math.sqrt(ms_err))+"\n")
    ((c, gamma), (acc, ms_err)) = max(res, key=lambda (c_gamma, (acc, ms_err)): acc)
    print("Lowest 0-1 loss (highest accuracy):\nC: "+str(c)+"\ngamma: "+str(gamma)+"\nAccuracy: "+str(acc)+"%\nRMS Error: "+str(math.sqrt(ms_err))+"\n")
    (t_pts, (t_acc, t_ms_err, t_corr), t_p_ests) = fun_eval((train_xs, test_xs), (train_ts, test_ts), c, gamma)
    print("\nFully trained SVM applied to test data:\nC: "+str(c)+"\ngamma: "+str(gamma)+"\nAccuracy: "+str(t_acc)+"%\nRMS Error: "+str(math.sqrt(t_ms_err))+"\n")

train_data = parse("../parkinsonsTrainStatML.dt")
test_data = parse("../parkinsonsTestStatML.dt")

print("\nRaw data:\n")
grid_search(train_data, test_data, raw_eval)

print("\nNormalized data:\n")
grid_search(train_data, test_data, norm_eval)

(t_xs, t_ts) = train_data
print("Training data:\n")
x = 1
for (mean, var) in map(mean_and_var, np.matrix(t_xs).T.A.tolist()):
    print("Column: "+str(x)+"\nMean: "+str(mean)+"\nVar: "+str(var)+"\n")
    x += 1
fun_norm = many_normalizer(np.matrix(t_xs))
(te_xs, te_ts) = test_data
n_te_xs = fun_norm(np.matrix(te_xs))
print("Normalized test data:\n")
x = 1
for (mean, var) in map(mean_and_var, n_te_xs.T.A.tolist()):
    print("Column: "+str(x)+"\nMean: "+str(mean)+"\nVar: "+str(var)+"\n")
    x += 1

# print(str(cross_validate(data, 5, 10, .1, norm_eval)))
# (t_acc, t_ms_err, t_n) = reduce(lambda t1, t2:
#                                 tuple(map(lambda x, y: x+y, t1, t2)),
#                                 res, (0, 0, 0))
# print(str ((t_acc/t_n, t_ms_err/t_n)) )

# (part_xs, part_ts) = tuple(map(partition(5), data))
# train_xs = part_xs[0][0]
# test_xs = part_xs[0][1]
# train_ts = part_ts[0][0]
# test_ts = part_ts[0][1]
# fun_norm = many_normalizer(np.matrix(train_xs))
# norm_train_xs = fun_norm(np.matrix(train_xs)).A.tolist()
# norm_test_xs = fun_norm(np.matrix(test_xs)).A.tolist()


# gammas = [.001, .01, .1, 1, 10]
# cs = [.1, 1, 10, 100, 1000]
# gamma = .1
# c = 10
# prob = svm.svm_problem(train_ts, norm_train_xs)
# param = svm.svm_parameter('-s '+str(svm.C_SVC)+' -t '+str(svm.RBF)+' -g '+str(gamma)+' -c '+str(c))
# model = svm.svm_train(prob, param)
# print(str(svm.svm_predict(test_ts, norm_test_xs, model)))


