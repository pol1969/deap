#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
from doc10 import * 
import numpy as np
import datetime as dt 
import pdb
import time


def test_getInitSchedule(setup10_docs):  
    sched = setup10_docs.getInitSchedule()
#    print(setup10_docs.getDocShifts(sched))


def test_getCorpus(setup10_docs):
    print([(i,setup10_docs.getCorpus(i)) for i in np.arange(1,len(setup10_docs)+1)])

def test_getDay(setup10_docs):
    print([(i,setup10_docs.getDay(i)) for i in np.arange(1,len(setup10_docs)+1)])


def test_isSuitableQuantity(setup10_docs):
    sched = np.zeros(len(setup10_docs),dtype=int)
    sched[[1,5,8,9,30,40]] = 23
    assert False == setup10_docs.isSuitableQuantity(sched,23,4)
    assert True == setup10_docs.isSuitableQuantity(sched,23,7)

def test_isSuitableSequence(setup10_docs):
    sched = np.empty(len(setup10_docs),None)
    sched[[1,5,8,15,30,40]] = 23
    assert False == setup10_docs.isSuitableSequence(sched,23,4,3)
    assert False == setup10_docs.isSuitableSequence(sched,23,29,3)
    assert True == setup10_docs.isSuitableSequence(sched,23,27,3)
    assert False == setup10_docs.isSuitableSequence(sched,23,41,3)

