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
    return (xs, map(lambda x: (x*2)-1, ts))

def mean_and_var(list):
    mean = sum(list)/len(list)
    vec_zero_mean = np.matrix(list).T - mean
    var = (vec_zero_mean.T * vec_zero_mean)/len(list)
    return (mean, var.A1[0])

def normalizer(list):
    (mean, var) = mean_and_var(list)
    return lambda new_list: map(lambda x: (x-mean)/math.sqrt(var), new_list)

# def apply(fun, list):
#     return fun(list)

def many_normalizer(mat):
    trans_dlist = mat.T.A
    return lambda new_mat: np.matrix(map(apply, map(normalizer, trans_dlist), new_mat.T.A)).T

def partition(folds):
    return lambda li: [([li[j] for j in range(len(li)) if j%folds != i], li[i::folds]) for i in range(folds)]

# test_li = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
# print(str(map(partition(5), (test_li, test_li[::-1]))))

def cross_validate(data, folds, c, gamma):
    (part_xs, part_ts) = tuple(map(partition(folds), data))

data = parse("../parkinsonsTrainStatML.dt")
(part_xs, part_ts) = tuple(map(partition(5), data))
train_xs = part_xs[0][0]
test_xs = part_xs[0][1]
train_ts = part_ts[0][0]
test_ts = part_ts[0][1]
fun_norm = many_normalizer(np.matrix(train_xs))
norm_train_xs = fun_norm(np.matrix(train_xs)).A.tolist()
norm_test_xs = fun_norm(np.matrix(test_xs)).A.tolist()


gammas = [.001, .01, .1, 1, 10]
cs = [.1, 1, 10, 100, 1000]
gamma = .1
c = 10
prob = svm.svm_problem(train_ts, norm_train_xs)
param = svm.svm_parameter('-s '+str(svm.C_SVC)+' -t '+str(svm.RBF)+' -g '+str(gamma)+' -c '+str(c))
model = svm.svm_train(prob, param)
print(str(svm.svm_predict(test_ts, norm_test_xs, model)))


