#! /usr/bin/env python
# -*- coding: utf-8 -*-

import math, string
import numpy as np
import matplotlib.pyplot as plot

class NeuralNetwork:
    """NeuralNetwork."""
    def __init__(self, in_D, hid_M, out_K, h, deriv_h, sigma=None, inv_sigma=None):
        self.in_D = in_D
        self.hid_M = hid_M
        self.out_K = out_K
        self.h = h
        self.deriv_h = deriv_h
        self.z = np.array([1.0]+([0.0]*(in_D+hid_M+out_K)))
        self.a = np.array([0.0]*hid_M)
        self.delta = np.array([0.0]*(hid_M+out_K))
        self.mask = np.matrix([[1 for j in range(1+in_D)]+[0 for j in range(hid_M+out_K-1)] for i in range(hid_M)]+[[1]+[0 for j in range(in_D)]+[1 for j in range(hid_M)]+[0 for j in range(out_K-1)] for i in range(out_K)])
        self.ws = np.matrix([[np.random.rand()*(2**-24) for j in range(in_D+hid_M+out_K)] for i in range(hid_M+out_K)])
        self.set_ws(self.ws)
        # wsli = [([np.random.rand()*(2**-24) for j in range(1+in_D+i)])+([.0]*(hid_M+out_K-i-1)) for i in range(hid_M+out_K)]
        # # for i in range(hid_M+out_K):
        # #     w = []
        # #     for j in range(1+in_D+hid_M+out_K):
        # #         if (j < 1+in_D+i):
        # #             w += [.01]
        # #         else:
        # #             w += [.0]
        # #     wsli += [w]
        # self.set_ws(np.matrix(wsli))
        self.grad = np.matrix([[0.0]*(in_D+hid_M+out_K)]*(hid_M+out_K))
        self.samples = 0
    def forward(self, xs):
        if (len(xs) != self.in_D):
            print("Error: NeuralNetwork forward wrong input length:\n len(xs)="+str(len(xs))+"\n in_D:"+str(self.in_D))
            return []
        for (x, i) in zip(xs, range(1, 1+self.in_D)):
            # print("i: "+str(i))
            self.z[i] = x
        for i in range(self.hid_M):
            iz = 1+self.in_D+i
            # print("i: "+str(iz))
            self.a[i] = self.z[:iz]*self.ws[i,:iz].T
            self.z[iz] = self.h(self.a[i])
        ts = []
        for i in range(self.hid_M, self.hid_M+self.out_K):
            iz = 1+self.in_D+i
            # print("i: "+str(iz))
            self.z[iz] = self.z[:iz]*self.ws[i,:iz].T
            ts += [self.z[iz]]
        return ts
    def backprop(self, ts):
        if (len(ts) != self.out_K):
            print("Error: NeuralNetwork backprop wrong output length:\n len(ts)="+str(len(ts))+"\n out_K:"+str(self.out_K))
        for i in range(self.out_K):
            iz = self.hid_M+self.out_K-i-1
            self.delta[iz] = self.z[1+self.in_D+iz]-ts[-i-1]
        for i in range(self.hid_M):
            iz = self.hid_M-i-1
            iw = 1+self.in_D+iz
            # print("iz: "+str(iz)+", iw: "+str(iw))
            self.delta[iz] = self.deriv_h(self.a[iz])*(self.delta[iz+1:]*self.ws[iz+1:,iw])
        # for (i, j) in [(a,b) for a in range(self.hid_M+self.out_K) for b in range(1+self.in_D+a)]:
        (i_dim, j_dim) = self.ws.shape
        for (i, j) in [(a, b) for a in range(i_dim) for b in range(j_dim) if self.mask[a, b] != 0]:
            # print("i: "+str(i)+", j: "+str(j))
            self.grad[i, j] += (self.delta[i]*self.z[j])
        self.samples += 1
        # self.grad += np.matrix(self.delta).T*np.matrix(self.z[1+self.in_D:])
    def gradient(self):
        return self.grad#/self.samples
    def get_mask(self):
        return self.mask
    def set_mask(self, mask):
        if (mask.shape != self.ws.shape):
            print("set_mask wrong shape!")
            return
        self.mask = mask
    def get_ws(self):
        return self.ws
    def set_ws(self, ws):
        if (ws.shape != self.ws.shape):
            print("set_ws wrong shape!")
            return
        (i_dim, j_dim) = ws.shape
        for (i, j) in [(a, b) for a in range(i_dim) for b in range(j_dim)]:
            self.ws[i, j] = self.mask[i, j]*ws[i, j]
        # for (i, j) in [(a,b) for a in range(self.hid_M+self.out_K) for b in range(1+self.in_D+a, self.in_D+self.hid_M+self.out_K)]:
        #     self.ws[i, j] = 0.0
    def grad_reset(self):
        self.grad = np.matrix([[0.0]*(self.in_D+self.hid_M+self.out_K)]*(self.hid_M+self.out_K))
        self.samples = 0
    def learn(self, eta):
        # print("ws: "+str(self.ws))
        # print("eta*grad: "+str(eta*self.grad))
        self.set_ws(self.ws-eta*self.gradient())



def s_error(ys, ts):
    err = np.matrix(ys)-np.matrix(ts)
    return .5*(err*err.T).A1[0]

def ms_error(ysli, tsli):
    return sum(map(s_error, ysli, tsli))#/len(ysli)

def train_pattern(network, xs, ts):
    ys = network.forward(xs)
    network.backprop(ts)
    return ys

e = 2**-29
def num_gradient(network, xs, ts):
    ys0 = map(lambda x: network.forward(x), xs)
    # print("ys0: "+str(ys0)+"\nts: "+str(ts))
    # return
    err0 = ms_error(ys0, ts)
    ws0 = network.get_ws()
    (i_dim, j_dim) = ws0.shape
    res = np.matrix([[.0]*j_dim]*i_dim)
    for (i,j) in [(a,b) for a in range(i_dim) for b in range(j_dim-(i_dim-a-1))]:
        print("i: "+str(i)+", j: "+str(j))
        ws = ws0.copy()
        ws[i, j] += e
        network.set_ws(ws)
        ys = map(lambda x: network.forward(x), xs)
        err = ms_error(ys, ts)
        res[i, j] = (err - err0)/e
    return res

delta_max = 2**0
delta_min = 2**-10
delta_init = 2**-8
def Rprop(network, train, count, eta_pos=1.2, eta_neg=.5, validate=None, plot_title=None, filename=None):
    if plot_title != None:
        plot.title(plot_title)
        plot.xlabel("iteration ->")
        plot.ylabel("ln(RMS) ->")
        plot.yscale('log')
    pxs = []
    trainRMSs = []
    if validate != None:
        valiRMSs = []
    network.grad_reset()
    grad_p = network.gradient()
    (i_dim, j_dim) = grad_p.shape
    delta_p = np.matrix([[np.random.rand()*delta_init for j in range(j_dim)] for i in range(i_dim)])
    for px in range(count):
        print("Iteration: "+str(px+1))
        if validate != None:
            ysli_vali = map(lambda xs: network.forward(xs), validate[0])
            valiRMSs += [math.sqrt(ms_error(ysli_vali, validate[1]))]
        ysli = map(lambda xs, ts: train_pattern(network, xs, ts), train[0], train[1])
        trainRMSs += [math.sqrt(ms_error(ysli, train[1]))]
        pxs += [px+1]
        grad_t = network.gradient()
        delta_t = delta_p.copy()
        ws = network.get_ws()
        for (i, j) in [(a, b) for a in range(i_dim) for b in range(j_dim)]:
            if grad_p[i, j]*grad_t[i, j] > 0:
                delta_t[i, j] = min(delta_p[i, j]*eta_pos, delta_max)
            else:
                delta_t[i, j] = max(delta_p[i, j]*eta_neg, delta_min)
            ws[i, j] -= np.sign(grad_t[i, j])*delta_t[i, j]
        # print("ws: "+str(ws-network.get_ws()))
        # print("delta_t: "+str(delta_t))
        network.set_ws(ws)
        network.grad_reset()
        grad_p = grad_t
        delta_p = delta_t
    if plot_title != None:
        plot.plot(pxs, trainRMSs, label="Train")
        if validate != None:
            plot.plot(pxs, valiRMSs, label="Validate")
        plot.legend()
        if (filename == None):
            plot.show()
        else:
            plot.savefig(filename)
    if validate != None:
        return (trainRMSs, valiRMSs)
    return trainRMSs

# def trainer(network, train, validate, plot_title, count, eta=.1, filename=None):
#     plot.title(plot_title)
#     plot.xlabel("iteration ->")
#     plot.ylabel("ln(RMS) ->")
#     plot.yscale('log')
#     pxs = []
#     trainRMSs = []
#     valiRMSs = []
#     # px = 1
#     # tsli = train[1]
#     for px in range(count):
#         print("Iteration: "+str(px+1))
#         # netw.apply_and_reset(eta)
#         ysli = map(lambda xs, ts: train_pattern(network, xs, ts), train[0], train[1])
#         # grad = network.gradient()
#         # new_ms_err = ms_error(ysli, tsli)
#         # d_ms_err = ms_err-new_ms_err
#         # ms_err = new_ms_err
#         # print("\nwhile: "+str(d_ms_err*(ms_err**errpow)/(samps**(1.0/smppow))))
#         # print("grad: "+str(netw.gradient()))
#         # print("ws[hidden]: "+str(netw.in_weights))
#         # print("ws[out]: "+str(netw.next.in_weights))
#         # print("ms_err: "+str(ms_err))
#         ysli_vali = map(lambda xs: network.forward(xs), validate[0])
#         trainRMSs += [math.sqrt(ms_error(ysli, train[1]))]
#         valiRMSs += [math.sqrt(ms_error(ysli_vali, validate[1]))]
#         pxs += [px+1]
#         # px += 1
#         network.learn(eta)
#         network.grad_reset()
#     plot.plot(pxs, trainRMSs, label="Train")
#     plot.plot(pxs, valiRMSs, label="Validate")
#     plot.legend()
#     if (filename == None):
#         plot.show()
#     else:
#         plot.savefig(filename)
#     # print("ArgMin(iteration) for validation RMS: "+str((min(zip(pxs, valiRMSs), key=lambda (it, rms): rms))))

def parse(filename):
    xs = []
    ts = []
    for line in open(filename).readlines():
        cols = string.split(line)
        if (len(cols) == 2):
            xs += [map(float, cols[:-1])]
            ts += [[float(cols[-1])]]
    return (xs, ts)


train = parse("../sincTrain25.dt")
validate = parse("../sincValidate10.dt")
# samples = map(lambda x: (x, (math.sin(x)/x)+np.random.normal(0, .004, 1)[0]), (np.random.rand(1000)*32).tolist())



activation_fun = lambda a: a/(1+abs(a))
deriv_activ_fun = lambda a: (1+abs(a))**-2

# network = NeuralNetwork(1, 20, 1, activation_fun, deriv_activ_fun)

# xs = map(lambda (x, t): [x], samples)
# ts = map(lambda (x, t): [t], samples)

# map(lambda x, t: train_pattern(network, x, t), xs, ts)
# back_grad = network.gradient()
# num_grad = num_gradient(network, xs, ts)
## ysli0 = map(lambda x, t: train_pattern(network, [x], [t]), xs, ts)
## err0 = ms_error(ysli0, tsli)

# def err_handler(type, flag):
#     print "Floating point error (%s), with flag %s" % (type, flag)

# saved_handler = np.seterrcall(err_handler)
# save_err = np.seterr(all='call')

network = NeuralNetwork(1, 20, 1, activation_fun, deriv_activ_fun)
# network.set_ws(np.matrix([[np.random.rand()*(2**-24),np.random.rand()*(2**-24),0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0] for i in range(20)]+[[0,0]+[np.random.rand()*(2**-24) for i in range(20)]]))
#trainer(network, train, validate, "Test", 100, .01, None)

Rprop(network, train, 200, 1.2, .5, validate, "Test")

