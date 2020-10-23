import pytest
from doc import * 
import numpy as np
import datetime as dt 
import pdb
import time

       

def test_all(setup_docs):
    # create a problem instance:
    
    assert type(setup_docs)==DocSchedulingProblem

def test_random_solution(setup_docs):
    randomSolution = np.random.randint(2, size=len(setup_docs))
 #   pdb.set_trace()
#    print("Random Solution = ")
 #   print(randomSolution)
#    print()
 #   print("Len randomSolution = ", len(randomSolution))
 
#    setup_docs.printScheduleInfo(randomSolution)
 
 #   print("Total Cost randomSolution = ",setup_docs.getCost(randomSolution))
 

@pytest.mark.skip()
def test_getInitShedule(setup_docs):
    start_time = time.time()
    print(time.time())
    myInitSolution = getInitShedule(setup_docs)
    print('Время выполнения,сек - ', time.time()-start_time)

#    print("My Init Solution= ")
#    print(myInitSolution)
 #   print("Len myInitSolution = ", len(myInitSolution))
 
#    setup_docs.printScheduleInfo(myInitSolution)
 
#    print("Total Cost myInitSolution = ", setup_docs.getCost(myInitSolution))

 
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


 #   print(ar1)

    c = np.array([1,0,0,1,1])

    c = np.where(c==0)
#    print(c)
    assert np.array_equal(ar1,c) 

def test_isSuitableCorpus(setup_docs):
    dejs = setup_docs.getRealDejs()
    days = setup_docs.getDaysInMonth()
    corps = setup_docs.getCorps()
    nmb_max = days*corps
    num_rows, num_cols  = dejs.shape
 #   print(dejs)
    for i in np.arange(num_rows):
        dej_doc = dejs.iloc[i]
        d =  np.random.randint(1,nmb_max+1)
  #      print(dej_doc['FAM'],d,isSuitableCorpus(setup_docs,i,d)) 
            

def test_assignToDej(setup_docs):
        
    schedule = np.zeros(len(setup_docs),dtype=np.int8)



def test_getNmbCorpusFrom1d():
    nmb_days = 30
    nmb_corps = 3

    assert 1 == getNmbCorpusFrom1d(20,nmb_days,nmb_corps)
    assert 2 == getNmbCorpusFrom1d(40,nmb_days,nmb_corps)
    assert 3 == getNmbCorpusFrom1d(70,nmb_days,nmb_corps)

def test_getDayFrom1d():
    nmb_days = 30
    nmb_corps = 3
    assert 2 == getDateFrom1d(3,nmb_days,nmb_corps)
    assert 1  == getDateFrom1d(32,nmb_days,nmb_corps)
#    assert 0 == getDateFrom1d(100,nmb_days,nmb_corps)
    assert 9 == getDateFrom1d(70,nmb_days,nmb_corps)
    assert 29 == getDateFrom1d(30,nmb_days,nmb_corps)




pytest.mark.skip()
def test_printScheduleHuman(setup_docs):
    schedule = np.zeros(len(setup_docs),dtype=np.int8)
    assignToDej(schedule,setup_docs,10,1,3)
#    assignToDej(schedule,setup_docs,2,30,3)
 #   assignToDej(schedule,setup_docs,3,30,2)

    printScheduleHuman(schedule, setup_docs)
    printScheduleHumanSum(schedule,setup_docs)
#    print(setup_docs.getRealDejs())

def test_isCorpRight():
    assert 1  == isCorpRight(100,1)
    assert 1  == isCorpRight(110,1)
    assert 1  == isCorpRight(101,1)
    assert 1  == isCorpRight(111,1)
    assert 1  == isCorpRight(10,2)
    assert 1  == isCorpRight(11,2)
    assert 1  == isCorpRight(110,2)
    assert 1  == isCorpRight(1,3)
    assert 1  == isCorpRight(11,3)
    assert 1  == isCorpRight(101,3)
    assert 0  == isCorpRight(100,2)

 
def test_isSuitableSequence(setup_docs):

    schedule = np.zeros(len(setup_docs),dtype=np.int8)
    assignToDej(schedule,setup_docs,0,5,1,1)
    assignToDej(schedule,setup_docs,0,10,1,1)
    assignToDej(schedule,setup_docs,0,15,2,1)
    assignToDej(schedule,setup_docs,0,2,5,3)
    
#    printScheduleHuman(schedule, setup_docs)

    assert False == isSuitableSequence(schedule, setup_docs,0,6)
    assert True == isSuitableSequence(schedule, setup_docs,0,1)
    assert False == isSuitableSequence(schedule, setup_docs,0,16)
    assert True == isSuitableSequence(schedule, setup_docs,0,30)
    assert False == isSuitableSequence(schedule, setup_docs,0,9)
    assert False == isSuitableSequence(schedule, setup_docs,0,31)
    assert True == isSuitableSequence(schedule, setup_docs,2,5)





 
def test_getAmbitOne(setup_docs):
    schedule = np.zeros(len(setup_docs),dtype=np.int8)
    assignToDej(schedule,setup_docs,0,5,1,1)
    assignToDej(schedule,setup_docs,0,9,1,1)
    assignToDej(schedule,setup_docs,0,25,2,1)

    dejs = setup_docs.getRealDejs()
    days = setup_docs.getDaysInMonth()
    corps = setup_docs.getCorps()
    nmb_max = days*corps
    num_rows, num_cols  = dejs.shape
    schedule = schedule.reshape(num_rows,nmb_max)
    schedule_doc = schedule[0]
    dej_doc = dejs.iloc[0]
#    print(dej_doc)
#    print(schedule_doc)
    neighb = getAmbitOne(schedule_doc,days,2)

def test_isSuitableQuantity_is_more_then_4(setup_docs):

    schedule = np.zeros(len(setup_docs),dtype=np.int8)
    assignToDej(schedule,setup_docs,0,1,1,1)
    assignToDej(schedule,setup_docs,0,5,1,1)
    assignToDej(schedule,setup_docs,0,10,1,1)
    assignToDej(schedule,setup_docs,0,15,1,1)
    assignToDej(schedule,setup_docs,0,20,1,1)

    assert False == isSuitableQuantity(schedule,setup_docs,0,4)



def test_isSuitableQuantity(setup_docs):

    schedule = np.zeros(len(setup_docs),dtype=np.int8)
    assignToDej(schedule,setup_docs,0,1,1,1)

    assert True == isSuitableQuantity(schedule,setup_docs,0,4)

def test_isScheduleFull_empty(setup_docs):
    schedule = np.zeros(len(setup_docs),dtype=np.int8)
    assert False == isScheduleFull(schedule,setup_docs)


def test_isScheduleFull_full(setup_docs):
    schedule = np.ones(len(setup_docs),dtype=np.int8)
    assert True == isScheduleFull(schedule,setup_docs)


def test_isScheduleFull_not_full(setup_docs):
    schedule = np.zeros(len(setup_docs),dtype=np.int8)
    assignToDej(schedule,setup_docs,0,1,1,1)
    assignToDej(schedule,setup_docs,0,5,1,1)
    assignToDej(schedule,setup_docs,0,10,1,1)
    assignToDej(schedule,setup_docs,0,15,1,1)
    assignToDej(schedule,setup_docs,0,20,1,1)
#    pdb.set_trace()
    assert False == isScheduleFull(schedule,setup_docs)


@pytest.mark.skip()
def test_convDayToDayAndCorp():
    days = 29 
    t = range(0,88)
    for i in t:
        print(i, convDayToDayAndCorp(i,days))

@pytest.mark.skip()
def test_isSuitableDej(setup_docs):
    schedule = np.zeros(len(setup_docs),dtype=np.int8)
    assignToDej(schedule,setup_docs,0,1,1,1)
    assignToDej(schedule,setup_docs,0,5,1,1)
    assignToDej(schedule,setup_docs,0,10,1,1)
    assignToDej(schedule,setup_docs,0,15,1,1)
    assignToDej(schedule,setup_docs,0,20,1,1)

 #   print(setup_docs.getRealDejs().shape[0])

 #   print(setup_docs.getRealDejs())
#    pdb.set_trace()
    assert False == isSuitableDej(schedule,setup_docs,0,20)
#    pdb.set_trace()
 #   assert True == isSuitableDej(schedule,setup_docs,2,65)
    
    assert False == isSuitableDej(schedule,setup_docs,4,20)


    printScheduleHuman(schedule,setup_docs)
    printScheduleHumanSum(schedule,setup_docs)
@pytest.mark.skip()    
def test_isFreeDay(setup_docs):
    schedule = np.zeros(len(setup_docs),dtype=np.int8)
    assignToDej(schedule,setup_docs,0,1,1,1)
    assignToDej(schedule,setup_docs,5,5,1,1)
    assignToDej(schedule,setup_docs,9,10,1,1)
    assignToDej(schedule,setup_docs,8,15,1,1)
    assignToDej(schedule,setup_docs,7,20,1,1)
    assignToDej(schedule,setup_docs,1,1,2,1)

#    printScheduleHuman(schedule,setup_docs)
#    printScheduleHumanSum(schedule,setup_docs)
 #   assert True == isFreeDay(schedule,setup_docs,3)
   # pdb.set_trace()
 #   assert True  == isFreeDay(schedule,5)
  #  assert True  == isFreeDay(schedule,20)
 #   assert False  == isFreeDay(schedule,10)
  #  assert True  == isFreeDay(schedule,2)
   # assert True  == isFreeDay(schedule,11)
  #  assert False  == isFreeDay(schedule,31)

    
def test_getDejForDoc(setup_docs):
    schedule = np.zeros(len(setup_docs),dtype=np.int8)
    assignToDej(schedule,setup_docs,0,1,1,1)
    assignToDej(schedule,setup_docs,0,5,1,1)
    assignToDej(schedule,setup_docs,0,9,1,1)
    assignToDej(schedule,setup_docs,5,5,2,1)
    assignToDej(schedule,setup_docs,9,10,3,1)
    assignToDej(schedule,setup_docs,8,15,2,1)
    assignToDej(schedule,setup_docs,7,20,1,1)

    getDejsForDoc(schedule,setup_docs,0)



@pytest.mark.skip()
def test_convDayCorpDejToFlatten(setup_docs):
    schedule = np.zeros(len(setup_docs),dtype=np.int8)
    z = convDayCorpDejToFlatten(schedule,setup_docs,1,1,4)
    schedule[z] = 1
    getDejsForDoc(schedule,setup_docs,0)
    printScheduleHuman(schedule,setup_docs)
    printScheduleHumanSum(schedule,setup_docs)
 

