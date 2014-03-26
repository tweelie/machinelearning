#! /usr/bin/env python
# -*- coding: utf-8 -*-

import math, string
import numpy as np
import matplotlib.pyplot as plot

def mult((x, y)):
    return x * y

def weighted_sum(vec_ins):
    return lambda vec_weights: sum(map(mult, filter(lambda (w, i): w > 0, zip(vec_weights, vec_ins))))

class NeuronLayer:
    """NeuronLayer."""
    def __init__(self, mat_in_weights, fun_activ=lambda x: x, fun_act_deriv=lambda x: 1, nl_next=None):
        self.in_weights = mat_in_weights
        self.activs = [0]*len(mat_in_weights)
        self.outs = [0]*len(mat_in_weights)
        self.ins = [0]*len(np.matrix(mat_in_weights).T.A.tolist())
        self.deltas = [0]*len(mat_in_weights)
        self.error_deriv_w = np.matrix([[.0]*len(self.ins)]*len(self.outs))
        self.samples = 0
        self.activ = fun_activ
        self.act_deriv = fun_act_deriv
        self.next = nl_next
        self.prev = None
        if (nl_next != None):
            nl_next.prev = self
    def gradient(self):
        if (self.next == None):
            return [self.error_deriv_w/self.samples]
        return [self.error_deriv_w/self.samples]+self.next.gradient()
    def apply_and_reset(self, eta):
        self.in_weights = (np.matrix(self.in_weights)-(eta*self.error_deriv_w/self.samples)).A.tolist()
        self.error_deriv_w = np.matrix([[.0]*len(self.ins)]*len(self.outs))
        self.samples = 0
        if(self.next != None):
            self.next.apply_and_reset(eta)
    def set_weights(self, weights):
        self.in_weights = weights[0]
        if (self.next != None):
            # print("ws: "+str(self.in_weights))
            self.next.set_weights(weights[1:])
    def forward(self, vec_ins):
        self.ins = [1]+vec_ins
        # print("Ins: "+str(self.ins))
        self.activs = map(weighted_sum(self.ins), self.in_weights)
        # print("Activs: "+str(self.activs))
        self.outs = map(self.activ, self.activs)
        # print("Outs: "+str(self.outs))
        if (self.next != None):
            return self.next.forward(self.outs)
        return self.outs
    def backprop_error(self, vec_targets):
        if(self.next != None):
            return self.next.backprop_error(vec_targets)
        # print("t: "+str(vec_targets))
        # print("outs(y): "+str(self.outs))
        # print("ins(z): "+str(self.ins))
        # print("(y[0]-t[0])*zs: "+str((self.outs[0]-vec_targets[0])*np.array(self.ins)))
        self.deltas = map(lambda y, t: y-t, self.outs, vec_targets)
        err_dw = np.matrix(map(lambda delta, zs: map(lambda z: delta*z, zs), self.deltas, [self.ins]*len(self.deltas)))
        # print("err_dw: "+str(err_dw))
        self.error_deriv_w += err_dw
        self.samples += 1
        if(self.prev != None):
            self.prev.backprop_error_hidden()
    def backprop_error_hidden(self):
        # print("xs: "+str(self.ins))
        # print("as: "+str(self.activs))
        # print("ws: "+str(self.in_weights))
        # print("next.deltas: "+str(self.next.deltas))
        self.deltas = map(mult, zip(map(self.act_deriv, self.activs), map(weighted_sum(self.next.deltas), np.matrix(self.in_weights).A.tolist())))
        # print("deltas: "+str(self.deltas))
        err_dw = np.matrix(map(lambda (delta, xs): map(lambda x: delta*x, xs), zip(self.deltas, [self.ins]*len(self.deltas))))
        # print("err_dw: "+str(err_dw))
        self.error_deriv_w += err_dw
        self.samples += 1
        if(self.prev != None):
            self.prev.backprop_error_hidden()


def init_network(weights, activation_funs, actfun_derivations):
    return reduce(lambda nl_next, (mat_in_weights, (fun_activ, fun_act_deriv)): NeuronLayer(mat_in_weights, fun_activ, fun_act_deriv, nl_next), zip(weights, zip(activation_funs, actfun_derivations))[-2::-1], NeuronLayer(weights[-1]))

def s_error(ys, ts):
    err = np.matrix(ys)-np.matrix(ts)
    return (err.T*err).A1[0]/2.0

def ms_error(ysli, tsli):
    return sum(map(s_error, ysli, tsli))/len(ysli)

def train_pattern(network, xs, ts):
    ys = network.forward(xs)
    network.backprop_error(ts)
    return ys

# hidden_ws = [[1, 1], [1, 1], [1, 1]]
hidden_ws = [[1, 1]]
hidden_activ = lambda a: a/(1+abs(a))
hidden_activ_deriv = lambda a: 1/math.pow(1+abs(a), 2)
# output_ws = [[1, 1, 1, 1]]
output_ws = [[1, 1]]
output_activ = id
output_activ_deriv = None
network = init_network([hidden_ws, output_ws], [hidden_activ, output_activ], [hidden_activ_deriv, output_activ_deriv])
samples = map(lambda x: (x, (math.sin(x)/x)+np.random.normal(0, .004, 1)[0]), (np.random.rand(1000)*32).tolist())

e = 2**-22
# hidden_e = [[0, 1],[2, 3],[4, 5]]
hidden_e = [[0, 1]]
# output_e = [[6, 7, 8, 9]]
output_e = [[2, 3]]
def e_or_zero(w, n_e):
    if(w == n_e):
        return e
    return .0
def e_setter(ws_li, n_e):
    return map(lambda ws: map(lambda w: e_or_zero(w, n_e), ws), ws_li)

ysli0 = map(lambda (x, t): train_pattern(network, [x], [t]), samples)
tsli = map(lambda (x, t): [t], samples)
err0 = ms_error(ysli0, tsli)
grad_bp = network.gradient()
# grad_num = [.0, .1, .2, .3, .4, .5, .6, .7, .8, .9]
grad_num = [.0, .1, .2, .3]
for i in range(4):
    network.apply_and_reset(1.0)
    network.set_weights([(np.matrix(hidden_ws)+np.matrix(e_setter(hidden_e, i))).A.tolist(),
                        (np.matrix(output_ws)+np.matrix(e_setter(output_e, i))).A.tolist()])
    yslii = map(lambda (x, t): train_pattern(network, [x], [t]), samples)
    erri = ms_error(yslii, tsli)
    grad_num[i] = (erri - err0)/e

print("Q III.1.1:\n")
print("Back-propagated gradient:")
print(str(grad_bp))
print("Numerically estimated gradient (e: "+str(e)+"):")
print(str(map(lambda mat: np.matrix(map(lambda vec: map(lambda i: grad_num[i], vec), mat)), [hidden_e, output_e])))

def parse(filename):
    xs = []
    ts = []
    for line in open(filename).readlines():
        cols = string.split(line)
        if (len(cols) == 2):
            xs += [map(float, cols[:-1])]
            ts += [float(cols[-1])]
    return (xs, ts)


train = parse("../sincTrain25.dt")
validate = parse("../sincValidate10.dt")

all_2_ws = [[[1, 1]]]*3
all_20_ws = [[[1, 1]]]*21

# grad_len = 100.0
# tmp = 32
def trainer(netw, plot_title, filename, eta= .1, n_while = .6):
    plot.title(plot_title)
    plot.xlabel("iteration ->")
    plot.ylabel("ln(RMS) ->")
    plot.yscale('log')
    pxs = []
    trainRMSs = []
    valiRMSs = []
    px = 1
    tsli = train[1]
    samps = len(train[0])
    errpow = 4.0
    smppow = 2.0
    ms_err = (2.0**1023)**(1/errpow)
    d_ms_err = 1.0 # 2.0**1023
    # ysli = map(lambda (x, t): train_pattern(netw, x, [t]), zip(train[0], train[1]))
    # new_ms_err = ms_error(ysli, tsli)
    # d_ms_err = ms_err-new_ms_err
    # ms_err = new_ms_err
    # print("\nwhile: "+str(d_ms_err*(ms_err**errpow)/(samps**(1.0/smppow))))
    # print("grad: "+str(netw.gradient()))
    # print("ws[hidden]: "+str(netw.in_weights))
    # print("ws[out]: "+str(netw.next.in_weights))
    # print("ms_err: "+str(ms_err))
    while(d_ms_err*(ms_err**errpow)/(samps**(1.0/smppow)) > n_while):
        # netw.apply_and_reset(eta)
        ysli = map(lambda (x, t): train_pattern(netw, x, [t]), zip(train[0], train[1]))
        # grad = network.gradient()
        new_ms_err = ms_error(ysli, tsli)
        d_ms_err = ms_err-new_ms_err
        ms_err = new_ms_err
        print("\nwhile: "+str(d_ms_err*(ms_err**errpow)/(samps**(1.0/smppow))))
        print("grad: "+str(netw.gradient()))
        # print("ws[hidden]: "+str(netw.in_weights))
        # print("ws[out]: "+str(netw.next.in_weights))
        print("ms_err: "+str(ms_err))
        ysli_vali = map(lambda xs: netw.forward(xs), validate[0])
        trainRMSs += [math.sqrt(ms_error(ysli, tsli))]
        valiRMSs += [math.sqrt(ms_error(ysli_vali, validate[1]))]
        pxs += [px]
        px += 1
        netw.apply_and_reset(eta)
    plot.plot(pxs, trainRMSs, label="Train")
    plot.plot(pxs, valiRMSs, label="Validate")
    plot.legend()
    plot.savefig(filename)
    print("ArgMin(iteration) for validation RMS: "+str((min(zip(pxs, valiRMSs), key=lambda (it, rms): rms))))

print("\nQ III.1.2:\n")
print("2 hidden neurons:\n")
network_2 = init_network(all_2_ws, ([hidden_activ]*2)+[output_activ], ([hidden_activ_deriv]*2)+[output_activ_deriv])
print("\nEta: 10.0")
trainer(network_2, "2 neurons, eta = 10", "img/n_2_eta_10.png", 10.0, 10**-12)
plot.figure(2)
network_2 = init_network(all_2_ws, ([hidden_activ]*2)+[output_activ], ([hidden_activ_deriv]*2)+[output_activ_deriv])
print("\nEta: 1.0")
trainer(network_2, "2 neurons, eta = 1", "img/n_2_eta_1.png", 1.0, 10**-12)
plot.figure(3)
network_2 = init_network(all_2_ws, ([hidden_activ]*2)+[output_activ], ([hidden_activ_deriv]*2)+[output_activ_deriv])
print("\nEta: 0.1")
trainer(network_2, "2 neurons, eta = .1", "img/n_2_eta_01.png", 0.1, 10**-12)
plot.figure(4)


network_2 = init_network(all_2_ws, ([hidden_activ]*2)+[output_activ], ([hidden_activ_deriv]*2)+[output_activ_deriv])
for i in range(20):
    map(lambda (x, t): train_pattern(network_2, x, [t]), zip(train[0], train[1]))
    network_2.apply_and_reset(.1)


print("\n20 hidden neurons:\n")
network_20 = init_network(all_20_ws, ([hidden_activ]*20)+[output_activ], ([hidden_activ_deriv]*20)+[output_activ_deriv])
print("\nEta: 10.0")
trainer(network_20, "20 neurons, eta = 10", "img/n_20_eta_10.png", 10.0, 10**-12)
plot.figure(5)
network_20 = init_network(all_20_ws, ([hidden_activ]*20)+[output_activ], ([hidden_activ_deriv]*20)+[output_activ_deriv])
print("\nEta: 1.0")
trainer(network_20, "20 neurons, eta = 1", "img/n_20_eta_1.png", 1.0, 10**-12)
plot.figure(6)
network_20 = init_network(all_20_ws, ([hidden_activ]*20)+[output_activ], ([hidden_activ_deriv]*20)+[output_activ_deriv])
print("\nEta: 0.1")
trainer(network_20, "20 neurons, eta = .1", "img/n_20_eta_01.png", 0.1, 10**-12)
plot.figure(7)
network_20 = init_network(all_20_ws, ([hidden_activ]*20)+[output_activ], ([hidden_activ_deriv]*20)+[output_activ_deriv])
print("\nEta: 0.01")
trainer(network_20, "20 neurons, eta = .01", "img/n_20_eta_001.png", 0.01, 5*10**-9)
plot.figure(8)

network_20 = init_network(all_20_ws, ([hidden_activ]*20)+[output_activ], ([hidden_activ_deriv]*20)+[output_activ_deriv])
for i in range(20):
    map(lambda (x, t): train_pattern(network_20, x, [t]), zip(train[0], train[1]))
    network_20.apply_and_reset(.1)

sinc = lambda x: math.sin(x)/x
net2 = lambda x: (network_2.forward([x]))[0]
net20 = lambda x: (network_20.forward([x]))[0]
x_range = [float(i)/20 for i in range(-200, 201) if i != 0]

plot.title("Trained network predictions compared with exact source model")
plot.xlabel("x ->")
plot.ylabel("f(x) ->")
plot.plot(x_range, map(sinc, x_range), label="sinc(x)")
plot.plot(x_range, map(net2, x_range), label="2 neurons")
plot.plot(x_range, map(net20, x_range), label="20 neurons")
plot.legend()
plot.savefig("img/nn.png")
plot.figure(9)


# network_20 = init_network([hidden_20_ws, output_20_ws], [hidden_activ, output_activ], [hidden_activ_deriv, output_activ_deriv])
# for i in range(14):
#     map(lambda (x, t): train_pattern(network_20, x, [t]), zip(train[0], train[1]))
#     network_20.apply_and_reset(.1)

# network_20 = init_network([hidden_20_ws, output_20_ws], [hidden_activ, output_activ], [hidden_activ_deriv, output_activ_deriv])


        # print("\nwhile: "+str(d_ms_err*(ms_err**errpow)/(samps**(1.0/smppow))))
        # print("\ngrad: "+str(grad))
        # print("ws[hidden]: "+str(network.in_weights))
        # print("ws[out]: "+str(network.next.in_weights))
        # print("ms_err: "+str(ms_err))
        # grad_len = sum(map(lambda (acc, n): acc/n, map(lambda mat: reduce(lambda (acc, n), vec: (acc+math.sqrt(sum((np.array(vec)**2).tolist())), n+1), mat.A.tolist(), (.0, 0)), grad)))/len(grad)
        # print(str(grad_len))
        # tmp -= 1

# map(lambda (x, t): train_pattern(network, [x], [t]), samples)
# print(str(network.gradient()))

# print(str(network.error_deriv_w/1000.0))
# print(str(network.next.error_deriv_w/1000.0))

# def train_testdata(network
