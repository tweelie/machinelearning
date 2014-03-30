#! /usr/bin/env python
# -*- coding: utf-8 -*-

import string
import numpy as np

# def s_error(ys, ts):
#     err = np.matrix(ys)-np.matrix(ts)
#     return .5*(err*err.T).A1[0]

# def ms_error(vec_ys, vec_ts):
#     return sum(map(s_error, vec_ys, vec_ts))/len(vec_ys)

def ms_error(mat_ys, mat_ts):
    return .5*np.matrix(np.array(mat_ys-mat_ts)**2).mean(0)

def parse(filename, ins=None, outs=1, outtype=float):
    xs = []
    ts = []
    for line in open(filename).readlines():
        cols = string.split(line)
        if ins == None:
            ins = len(cols)-outs
        if (len(cols) == ins+outs):
            if outs == 0:
                xs += [map(float, cols)]
            else:
                xs += [map(float, cols[:-outs])]
                ts += [map(outtype, cols[-outs:])]
    return (np.matrix(xs), np.matrix(ts))


def mean_and_stddev(mat_train):
    mean = mat_train.mean(0)
    stddev = mat_train.std(0)
    # var = stddev**2
    return (mean, stddev)

def normalizer(mat_train):
    (mean, stddev) = mean_and_stddev(mat_train)
    return lambda mat_other: (mat_other-mean)/stddev

# def apply(fun, list):
#     return fun(list)

# def __multi_normalizer_inner(mat_train, mat_other):
#     trans_dlist = mat_train.T.A
#     return np.matrix(map(lambda fn, el: apply(fn, [el]), map(normalizer, trans_dlist), mat_other.T.A)).T

# def multi_normalizer(mat_train):
#     return lambda mat_other: __multi_normalizer_inner(mat_train, mat_other)

def __partition_inner(folds, li):
    return ([np.matrix([li[j] for j in range(len(li)) if j%folds != i]) for i in range(folds)], [np.matrix(li[i::folds]) for i in range(folds)])

def partition(folds):
    return lambda mat: __partition_inner(folds, mat.tolist())

# fun_eval(train, test): res
def cross_validate(data, folds, fun_eval):
    ((tr_xs, te_xs), (tr_ts, te_ts)) = tuple(map(partition(folds), data))
    return map(fun_eval, zip(tr_xs, tr_ts), zip(te_xs, te_ts))


