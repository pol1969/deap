import pytest
from doc import DocSchedulingProblem,getInitShedule
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
    myInitSolution = getInitShedule(setup_docs,dt.datetime.now().month+1,dt.datetime.now().year)

    print("My Init Solution= ")
 #   print(myInitSolution)
    print()
    print("Len myInitSolution = ", len(myInitSolution))
 
#    setup_docs.printScheduleInfo(myInitSolution)
 
    print("Total Cost myInitSolution = ", setup_docs.getCost(myInitSolution))
 

