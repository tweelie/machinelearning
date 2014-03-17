#! /usr/bin/env python
# -*- coding: utf-8 -*-

import string

def parse(filename):
    res = []
    for line in open(filename).readlines():
        cols = string.split(line)
        res += [map(float, cols)]
    return res

