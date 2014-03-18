#! /usr/bin/env python
# -*- coding: utf-8 -*-

import math
import numpy as np

def mult((x, y)):
    return x * y

def weighted_sum(vec_ins):
    return lambda vec_weights: sum(map(mult, filter(lambda (w, i): w > 0, zip(vec_weights, vec_ins))))

class NeuronLayer:
    """NeuronLayer."""
    def __init__(self, mat_in_weights, fun_activ=id, fun_act_deriv=lambda x: 1, nl_next=None):
        self.in_weights = mat_in_weights
        self.activs = [0]*len(mat_in_weights)
        self.outs = [0]*len(mat_in_weights)
        self.ins = [0]*len(np.matrix(mat_in_weights).T.A.tolist())
        self.deltas = [0]*len(mat_in_weights)
        self.error_deriv_w = np.matrix([[0]*len(self.ins)]*len(self.outs))
        self.activ = fun_activ
        self.act_deriv = fun_act_deriv
        self.next = nl_next
        self.prev = None
        if (nl_next != None):
            nl_next.prev = self
    def zero_error_deriv(self):
        self.error_deriv_w = np.matrix([[0]*len(self.ins)]*len(self.outs))
    def forward(self, vec_ins):
        self.ins = [1]+vec_ins
        self.activs = map(weighted_sum(self.ins), self.in_weights)
        self.outs = map(self.activ, self.activs)
        if (self.next != None):
            return self.next.forward(self.outs)
        return self.outs
    def backprop_error(self, vec_targets):
        if(self.next != None):
            self.next.backprop_error(vec_targets)
        self.deltas = map(lambda (y, t): y-t, zip(self.outs, vec_targets))
        err_dw = np.matrix(map(lambda (delta, zs): map(lambda z: delta*z, zs), zip(self.deltas, [self.ins]*len(self.deltas))))
        self.error_deriv_w += err_dw
        if(self.prev != None):
            self.prev.backprop_error_hidden()
    def backprop_error_hidden(self):
        self.deltas = map(mult, zip(map(self.act_deriv, self.outs), map(weighted_sum(self.next.deltas), np.matrix(self.next.in_weights).T.A.tolist())))
        self.error_deriv_w += np.matrix(map(lambda (delta, zs): map(lambda z: delta*z, zs), zip(self.deltas, [self.ins]*len(self.deltas))))
        if(self.prev != None):
            self.prev.backprop_error_hidden()


def init_network(weights, activation_funs, actfun_derivations):
    return reduce(lambda nl_next, (mat_in_weights, (fun_activ, fun_act_deriv)): NeuronLayer(mat_in_weights, fun_activ, fun_act_deriv, nl_next), zip(weights, zip(activation_funs, actfun_derivations))[-2::-1], NeuronLayer(weights[-1]))

def train_pattern(network, xs, ts):
    network.forward(xs)
    network.backprop_error(ts)

hidden_ws = [[1, 1], [1, 1], [1, 1]]
hidden_activ = lambda a: a/(1+abs(a))
hidden_activ_deriv = lambda a: 1/math.pow(1+abs(a), 2)
output_ws = [[1, 1, 1, 1]]
output_activ = id
output_activ_deriv = None
network = init_network([hidden_ws, output_ws], [hidden_activ, output_activ], [hidden_activ_deriv, output_activ_deriv])
samples = map(lambda x: (x, (math.sin(x)/x)+np.random.normal(0, .004, 1)[0]), (np.random.rand(1000)*32).tolist())
map(lambda (x, t): train_pattern(network, [x], [t]), samples)
print(str(network.error_deriv_w))
print(str(network.next.error_deriv_w))

# def train_testdata(network
