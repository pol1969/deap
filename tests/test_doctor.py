import pytest
from doc import * 
import numpy as np
import datetime as dt 
import pdb

       

def test_all(setup_docs):
    # create a problem instance:
    
    assert type(setup_docs)==DocSchedulingProblem

def test_random_solution(setup_docs):
    randomSolution = np.random.randint(2, size=len(setup_docs))
 #   pdb.set_trace()
    print("Random Solution = ")
 #   print(randomSolution)
    print()
    print("Len randomSolution = ", len(randomSolution))
 
#    setup_docs.printScheduleInfo(randomSolution)
 
    print("Total Cost randomSolution = ",setup_docs.getCost(randomSolution))
 


def test_getInitShedule(setup_docs):
    myInitSolution = getInitShedule(setup_docs)
    print()

    print("My Init Solution= ")
#    print(myInitSolution)
    print("Len myInitSolution = ", len(myInitSolution))
 
#    setup_docs.printScheduleInfo(myInitSolution)
 
    print("Total Cost myInitSolution = ", setup_docs.getCost(myInitSolution))
 
def test_getRealDejs(setup_docs):
    assert len(setup_docs.getDfDocs())>len(setup_docs.getRealDejs())


def test_getFreeDejs():
    nmb_corps = 1
    nmb_dej = 2
    days_in_month = 5 
    ar =np.zeros(10,dtype=int)
    for idx,a in enumerate(ar):
        if idx % 4 == 0:
            ar[idx]=1


    ar1 = getFreeDejFromSchedule(ar,days_in_month,nmb_corps)


    print(ar1)

    c = np.array([1,0,0,1,1])

    c = np.where(c==0)
    print(c)
    assert np.array_equal(ar1,c) 

    
