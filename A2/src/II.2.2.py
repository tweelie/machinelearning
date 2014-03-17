#! /usr/bin/env python
# -*- coding: utf-8 -*-

import math, parser
import numpy as np
import matplotlib.pyplot as plot

# import II.1.1-2

beta = 1.0

def eq3_54(alpha, beta, mat_phi):
    mat_sigma_N_inv = alpha * np.identity(len(mat_phi.T)) + beta * mat_phi.T * mat_phi
    return mat_sigma_N_inv

def eq3_53(beta, mat_sigma_N, mat_phi, vec_t):
    vec_m_N = beta * mat_sigma_N * mat_phi.T * np.asmatrix(vec_t).T
    return vec_m_N

def y(vec_w):
    return lambda li_x: (np.asmatrix(vec_w) * np.asmatrix([1.0] + li_x).T).A1[0]

def RMS(y, xs, ts):
    return math.sqrt(((np.array(map(y, xs)) - np.array(ts))**2).mean())
def RMS_it(y, xst_li):
    return math.sqrt(((np.array(map(lambda xst: y(xst[:-1]), xst_li)) - np.array(map(lambda xst: xst[-1], xst_li)))**2).mean())

testdata = parser.parse("../sunspotsTestStatML.dt")
traindata = parser.parse("../sunspotsTrainStatML.dt")

__concat1 = lambda li: [1.0] + li

sel1 = lambda data: map(lambda col: col[2:4], data)
sel1_it = lambda data: map(lambda col: col[2:4] + [col[5]], data)
sel2 = lambda data: map(lambda col: [col[4]], data)
sel2_it = lambda data: map(lambda col: [col[4], col[5]], data)
sel3 = lambda data: map(lambda col: col[0:5], data)
sel3_it = id
def ts(data):
    return map(lambda col: col[5], data)

def prior1(alpha, xst):
    mat_phi = np.matrix([1.0] + xst[:-1])
    mat_sigma_N_inv = alpha*np.identity(len(xst)) + beta*mat_phi.T*mat_phi
    vec_m_N = beta*mat_sigma_N_inv.getI()*mat_phi.T*xst[-1]
    return (vec_m_N, mat_sigma_N_inv)

def iterate((vec_m_0, mat_sigma_0_inv), xst):
    mat_phi = np.matrix([1.0] + xst[:-1])
    mat_sigma_N_inv = mat_sigma_0_inv + beta*mat_phi.T*mat_phi
    vec_m_N = mat_sigma_N_inv.getI()*(mat_sigma_0_inv*vec_m_0 + beta*mat_phi.T*xst[-1])
    return (vec_m_N, mat_sigma_N_inv)

def apply_it(alpha, sel):
    li_xst = sel(traindata)
    (w_map, nocare) = reduce(iterate, li_xst[1:], prior1(alpha, li_xst[0]))
    return RMS_it(y(w_map.A1), sel(testdata))

# reduce(lambda (x0, x1), y: (x0+y, x1*y), [1, 2, 3, 4, 5], (0, 1))

def apply(alpha, sel):
    mat_phi = np.matrix(map(__concat1, sel(traindata)))
    mat_sigma_N = eq3_54(alpha, beta, mat_phi).getI()
    w_map = eq3_53(beta, mat_sigma_N, mat_phi, np.array(ts(traindata))).A.flatten()
    return RMS(y(w_map), sel(testdata), ts(testdata))

def applyH(sel):
    return lambda alpha: apply(alpha, sel)
def apply_itH(sel):
    return lambda alpha: apply_it(alpha, sel)

def plot_alpha_rg(title, filename, alphas, sel):
    plot.title(title)
    plot.xlabel("alpha ->")
    plot.ylabel("ln(RMS) ->")
    plot.yscale('log')
    RMSs = map(applyH(sel), alphas)
    plot.plot(alphas, RMSs)
    minRMS = min(zip(alphas, RMSs), key=lambda (a, rms): rms)
    print "Min(RMS) for " + title + ": " + str(minRMS[1]) + " for alpha: " + str(minRMS[0])
    plot.savefig(filename)

alphas = range(-100, 100)
plot_alpha_rg("Selection 1", "img/sel1.png", alphas, sel1)
plot.figure(2)
plot_alpha_rg("Selection 2", "img/sel2.png", alphas, sel2)
plot.figure(3)
plot_alpha_rg("Selection 3", "img/sel3.png", alphas, sel3)
plot.figure(4)
plot.title("All selections")
plot.xlabel("alpha ->")
plot.ylabel("ln(RMS) ->")
plot.yscale('log')
plot.plot(alphas, map(applyH(sel1), alphas))
plot.plot(alphas, map(applyH(sel2), alphas))
plot.plot(alphas, map(applyH(sel3), alphas))
plot.plot(alphas, [35.4650589931] * len(alphas))
plot.plot(alphas, [28.8397676572] * len(alphas))
plot.plot(alphas, [18.7700074768] * len(alphas))
plot.savefig("img/selall.png")


