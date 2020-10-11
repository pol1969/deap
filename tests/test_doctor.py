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

def test_isSuitableCorpus(setup_docs):
    df = setup_docs.getRealDejs()
    schedule = np.zeros(len(setup_docs),dtype=np.int8)
    """
    print(len(schedule))
    print(schedule)
    print()
    print(df.iloc[0]['FAM'])

    print(df.iloc[0]['CORPUS'])
    """

def test_assignToDej(setup_docs):
        
    schedule = np.zeros(len(setup_docs),dtype=np.int8)
    assignToDej(schedule,setup_docs,5,5,4,1)


def test_getNmbCorpusFrom1d():
    nmb_days = 30
    nmb_corps = 3

    assert 1 == getNmbCorpusFrom1d(20,nmb_days,nmb_corps)
    assert 2 == getNmbCorpusFrom1d(40,nmb_days,nmb_corps)
    assert 3 == getNmbCorpusFrom1d(70,nmb_days,nmb_corps)
    assert 0 == getNmbCorpusFrom1d(100,nmb_days,nmb_corps)

def test_getDayFrom1d():
    nmb_days = 30
    nmb_corps = 3
    assert 3 == getDateFrom1d(3,nmb_days,nmb_corps)
    assert 2  == getDateFrom1d(32,nmb_days,nmb_corps)
    assert 0 == getDateFrom1d(100,nmb_days,nmb_corps)
    assert 10 == getDateFrom1d(70,nmb_days,nmb_corps)








def test_printScheduleHuman(setup_docs):
    schedule = np.zeros(len(setup_docs),dtype=np.int8)
    assignToDej(schedule,setup_docs,4,1,1,1)
    assignToDej(schedule,setup_docs,5,2,2,1)
    assignToDej(schedule,setup_docs,8,3,3,1)
    assignToDej(schedule,setup_docs,1,5,1,1)
    assignToDej(schedule,setup_docs,2,2,1,1)

    printScheduleHuman(schedule, setup_docs)




