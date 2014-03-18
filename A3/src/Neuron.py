#! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

def __mult(x, y):
    return x * y

def __weighted_sum(vec_ins):
    lambda vec_weights: sum(map(__mult, filter(lambda (w, i): w > 0, zip(vec_weights, vec_ins))))

class NeuronLayer:
    """NeuronLayer."""
    def __init__(self, mat_in_weights, fun_activ=id, fun_act_deriv=lambda x: 1, nl_next=None):
        self.in_weights = mat_in_weights
        self.activs = [0]*len(mat_in_weights)
        self.outs = [0]*len(mat_in_weights)
        self.ins = [0]*len(np.matrix(mat_in_weights).T.A2)
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
        self.activs = map(__weighted_sum(vec_ins), self.in_weights)
        self.outs = map(self.activ, self.activs)
        if (self.next != None):
            return self.next.forward(self.outs)
        return self.outs
    def backprop_error(self, vec_targets):
        self.deltas = map(lambda (y, t): y-t, zip(self.outs, vec_targets))
        self.error_deriv_w += np.matrix(map(lambda (delta, zs): map(lambda z: delta*z, zs), zip(self.deltas, [self.ins]*len(self.deltas))))
        if(self.prev != None):
            self.prev.backprop_error()
    def backprop_error(self):
        self.deltas = map(__mult, zip(map(self.act_deriv, self.outs), map(__weighted_sum(next.deltas), np.matrix(next.in_weights).T.A2)))
        self.error_deriv_w += np.matrix(map(lambda (delta, zs): map(lambda z: delta*z, zs), zip(self.deltas, [self.ins]*len(self.deltas))))
        if(self.prev != None):
            self.prev.backprop_error()


def init_network(weights, activation_funs, actfun_derivations):
    return reduce(lambda nl_next, (mat_in_weights, (fun_activ, fun_act_deriv)): NeuronLayer(mat_in_weights, fun_activ, fun_act_deriv, nl_next), zip(weights, zip(activation_funs, actfun_derivations))[:0:-1], NeuronLayer(weights[-1]))

def train_pattern(network, xs, ts):
    network.forward(xs)
    network.backprop_error(ts)

# def train_testdata(network