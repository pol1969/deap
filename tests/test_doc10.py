#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
from doc10 import * 
import numpy as np
import datetime as dt 
import pdb
import time
import os.path


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

def test_isSuitableWish(setup10_docs):

    if os.path.exists('wish.csv'):
        print('Файл wish.csv существует, читаем ...')
        dw = pd.read_csv('wish.csv')
    else:
        print('Файл wish.csv не существует, создаем ...')
        
        dw = pd.read_csv('lk_1.csv',usecols=[5])
        arrays = [ np.random.choice(np.arange(1,32),10,
            replace=False) for _ in range(dw.shape[0])]

        dw['NO'] = arrays
        dw['Month'] = setup10_docs.getMonth()
        dw['Year'] = setup10_docs.getYear()
        dw.to_csv('wish.csv')

    print(dw)
    ar = dw.loc[38,'NO']

    if isinstance(dw.loc[38,'NO'],str):
        print('str')
        s = (dw.loc[38,'NO']).replace('[','')
        s = (s).replace(']','')
        ar = np.array(s.split()).astype(int)
    else:
        print('not str')

    print(ar)



