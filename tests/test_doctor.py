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
 

pytest.mark.skip()
def test_getInitSchedule(setup_docs):
    start_time = time.time()
    print(time.time())
    myInitSolution = getInitSchedule(setup_docs)
    print('Время выполнения,сек - ', time.time()-start_time)
    return myInitSolution

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

    days = setup_docs.getDaysInMonth()
    corps = setup_docs.getCorps()
    dejs = setup_docs.getNmbRealDejs()
    docs = setup_docs.getRealDejs()

        
#    schedule = np.zeros(len(setup_docs),dtype=np.int8)
    
    cnt = 10
    while cnt >=0:
        day = np.random.randint(1,days+1)
        dej = np.random.randint(1,dejs)
        corp = np.random.randint(1,corps+1)
 #       pdb.set_trace()
        day_in_sched = convDejDayCorpToFlatten(setup_docs,
                dej,day,corp)
        
 #       print(docs.iloc[dej]['FAM'], docs.iloc[dej]['CORPUS'],
#                day_in_sched,'corp=',corp,
 #               isSuitableCorpus(setup_docs,dej,day_in_sched)) 
        assert isCorpRight(docs.iloc[dej]['CORPUS'],corp) == isSuitableCorpus(setup_docs,dej,day_in_sched)
        cnt -= 1






def test_getNmbCorpusFrom1d():
    nmb_days = 30
    nmb_corps = 3

    assert 1 == getNmbCorpusFrom1d(20,nmb_days,nmb_corps)
    assert 2 == getNmbCorpusFrom1d(40,nmb_days,nmb_corps)
    assert 3 == getNmbCorpusFrom1d(70,nmb_days,nmb_corps)
    assert 1 == getNmbCorpusFrom1d(29,nmb_days,nmb_corps)


def test_getDayFrom1d():
    nmb_days = 30
    nmb_corps = 3
    assert 3 == getDateFrom1d(3,nmb_days,nmb_corps)
    assert 2  == getDateFrom1d(32,nmb_days,nmb_corps)
#    assert 0 == getDateFrom1d(100,nmb_days,nmb_corps)
    assert 10 == getDateFrom1d(70,nmb_days,nmb_corps)
 #   assert 29 == getDateFrom1d(30,nmb_days,nmb_corps)




pytest.mark.skip()
def test_printScheduleHuman(setup_docs):
    schedule = np.zeros(len(setup_docs),dtype=np.int8)
    assignToDej(schedule,setup_docs,0,1,1)
    assignToDej(schedule,setup_docs,2,1,2)
    assignToDej(schedule,setup_docs,3,1,3)
    assignToDej(schedule,setup_docs,4,30,1)
    assignToDej(schedule,setup_docs,5,30,2)
    assignToDej(schedule,setup_docs,6,30,3)

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
    days = setup_docs.getDaysInMonth()
    corps = setup_docs.getCorps()
    dejs = setup_docs.getNmbRealDejs()
    nmb_neighb=3


        
    schedule = np.zeros(len(setup_docs),dtype=np.int8)
    
    cnt = 10
    while cnt >=0:
        day = np.random.randint(1,days+1)
        dej = np.random.randint(1,dejs)
        corp = np.random.randint(1,corps+1)
        assignToDej(schedule,setup_docs,dej,day,corp)
 #       print(np.where(schedule==1))
        i = convDejDayCorpToFlatten(setup_docs,dej,day,corp)
        print('День ', day)
        
        j = i + np.random.randint(1,nmb_neighb+1)
        if j > days-1:
            j=i
        print('j+',j)


        assert False == isSuitableSequence(schedule,setup_docs,j)

 #       assert True == isSuitableSequence(schedule,setup_docs,j+5)



 #       pdb.set_trace()
 #       j = i - np.random.randint(1,nmb_neighb+1)
  #      if j <= 0:
   #         j=i
    #    print('j-',j)


#        assert False == isSuitableSequence(schedule,setup_docs,j)


        assignToDej(schedule,setup_docs,dej,day,corp,0)



        cnt -= 1




 
def test_getAmbitOne(setup_docs):
    schedule = np.zeros(len(setup_docs),dtype=np.int8)
    assignToDej(schedule,setup_docs,0,1,1,1)
    assignToDej(schedule,setup_docs,0,9,1,1)
    assignToDej(schedule,setup_docs,0,25,2,1)

    neighb = getAmbitOne(schedule,setup_docs,0,2)
    print(neighb)
    assert True == np.array_equal(neighb,[1,2,3,7,8,9,10,11,23,24,25,26,27])

def test_getAmbitOne_right_margin(setup_docs):
    schedule = np.zeros(len(setup_docs),dtype=np.int8)
    days_in_month = setup_docs.getDaysInMonth()
    nmb_neighb = 2
    i = 29
    max_int = nmb_neighb*2+1
    min_margin = i - nmb_neighb
    max_margin = min_margin + max_int
    if max_margin > days_in_month:
        max_margin = days_in_month

    ar = np.arange(min_margin,max_margin+1)

    assignToDej(schedule,setup_docs,0,29,1,1)

    neighb = getAmbitOne(schedule,setup_docs,0,nmb_neighb)
    print(ar)
    print(neighb)

    assert True == np.array_equal(neighb,ar)


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



pytest.mark.skip()
def test_isSuitableDej_on_corpus(setup_docs):
    schedule = np.zeros(len(setup_docs),dtype=np.int8)
    assignToDej(schedule,setup_docs,0,1,1,1)
    assignToDej(schedule,setup_docs,0,5,1,1)
    assignToDej(schedule,setup_docs,0,20,1,1)

    #не тот корпус 2
    i = convDejDayCorpToFlatten(setup_docs,0,29,2)
#    pdb.set_trace()
    assert False == isSuitableDej(schedule,setup_docs,i,4)
  
    #не тот корпус 3
    i = convDejDayCorpToFlatten(setup_docs,0,29,3)
    assert False == isSuitableDej(schedule,setup_docs,i,3)

    

pytest.mark.skip()
def test_isSuitableDej_on_quantity(setup_docs):
    schedule = np.zeros(len(setup_docs),dtype=np.int8)
    assignToDej(schedule,setup_docs,0,1,1,1)
    assignToDej(schedule,setup_docs,0,5,1,1)
    assignToDej(schedule,setup_docs,0,20,1,1)
    assignToDej(schedule,setup_docs,0,25,1,1)


    # лишнее дежурство 
    i = convDejDayCorpToFlatten(setup_docs,0,29,2)
#    pdb.set_trace()
    assert False == isSuitableDej(schedule,setup_docs,i,3)



pytest.mark.skip()
def test_isSuitableDej_on_sequence(setup_docs):
    schedule = np.zeros(len(setup_docs),dtype=np.int8)
    assignToDej(schedule,setup_docs,0,1,1,1)
    assignToDej(schedule,setup_docs,0,5,1,1)
    assignToDej(schedule,setup_docs,0,20,1,1)


    # смежное дежурство  
    i = convDejDayCorpToFlatten(setup_docs,0,6,1)
    assert False == isSuitableDej(schedule,setup_docs,i,3)

    # смежное дежурство  с перерывом в 1 день 
    i = convDejDayCorpToFlatten(setup_docs,0,7,1)
    assert False == isSuitableDej(schedule,setup_docs,i,3)

    # смежное дежурство  с перерывом в 2 дня 
    i = convDejDayCorpToFlatten(setup_docs,0,8,1)
    assert False == isSuitableDej(schedule,setup_docs,i,3)


pytest.mark.skip()
def test_isSuitableDej_on_sequence_diff_corpus(setup_docs):
    schedule = np.zeros(len(setup_docs),dtype=np.int8)
    assignToDej(schedule,setup_docs,15,1,1,1)
    assignToDej(schedule,setup_docs,15,5,1,1)

    #смежное дежурство на разных корпусах
    i = convDejDayCorpToFlatten(setup_docs,15,2,2)
    assert False == isSuitableDej(schedule,setup_docs,i,3)

    #одинаковое  дежурство на разных корпусах
    i = convDejDayCorpToFlatten(setup_docs,15,1,2)
    assert False == isSuitableDej(schedule,setup_docs,i,3)







pytest.mark.skip()    
def test_isFreeDay(setup_docs):
  
    days = setup_docs.getDaysInMonth()
    corps = setup_docs.getCorps()
    dejs = setup_docs.getNmbRealDejs()

        
    schedule = np.zeros(len(setup_docs),dtype=np.int8)
    assignToDej(schedule,setup_docs,0,1,1)
    assignToDej(schedule,setup_docs,0,30,1)
    assignToDej(schedule,setup_docs,5,15,3)

    i = convDejDayCorpToFlatten(setup_docs,0,1,1)

    i1 = convDejDayCorpToFlatten(setup_docs,1,1,1)
   
    i2 = convDejDayCorpToFlatten(setup_docs,1,30,1)
   
    i3 = convDejDayCorpToFlatten(setup_docs,1,15,1) 
    
    i4 = convDejDayCorpToFlatten(setup_docs,3,15,3) 



    assert False == isFreeDay(schedule,setup_docs,i) 

    assert False == isFreeDay(schedule,setup_docs,i1) 
    
    assert True == isFreeDay(schedule,setup_docs,i3) 
#    pdb.set_trace() 
    assert False == isFreeDay(schedule,setup_docs,i4) 

    



    
def test_getDejForDoc(setup_docs):
    schedule = np.zeros(len(setup_docs),dtype=np.int8)
    ar = np.array([[1,1],[5,1],[9,3],[25,2],[20,2],[28,3]])
    ir = np.empty(0,dtype=int)
    for i in ar:
        assignToDej(schedule,setup_docs,0,i[0],i[1],1)
        ir = np.append(ir,convDejDayCorpToFlatten(setup_docs,
            0,i[0],i[1])+1)
    


    r = getDejsForDoc(schedule,setup_docs,0)
    ir = np.sort(ir)
    r = np.sort(r)
    print(ir)
    print(r)


    assert True == np.array_equal(r,ir)




pytest.mark.skip()
def test_convDejDayCorpToFlatten(setup_docs):
    """
    Первый по-настоящемк хороший железный тест
    через разработку. Нашел формулу перевода доктор-день-корпус
    в индекс schedule через assignToDej
    """

    days = setup_docs.getDaysInMonth()
    corps = setup_docs.getCorps()
    dejs = setup_docs.getNmbRealDejs()

        
    schedule = np.zeros(len(setup_docs),dtype=np.int8)
    
    cnt = 10
    while cnt >=0:
        day = np.random.randint(1,days+1)
        dej = np.random.randint(1,dejs)
        corp = np.random.randint(1,corps+1)
        assignToDej(schedule,setup_docs,dej,day,corp)
 #       print(np.where(schedule==1))
        i = convDejDayCorpToFlatten(setup_docs,dej,day,corp)
  #      print(i)

        assert schedule[i]==1
        cnt -= 1



def test_convFlattenToDejDayCorp(setup_docs):
    """
    Первый по-настоящемк хороший железный тест
    через разработку. Нашел формулу перевода доктор-день-корпус
    в индекс schedule через assignToDej
    """

    days = setup_docs.getDaysInMonth()
    corps = setup_docs.getCorps()
    dejs = setup_docs.getNmbRealDejs()

        
    schedule = np.zeros(len(setup_docs),dtype=np.int8)
    
    cnt = 10
    while cnt >=0:
        day = np.random.randint(1,days+1)
        dej = np.random.randint(1,dejs)
        corp = np.random.randint(1,corps+1)
        assignToDej(schedule,setup_docs,dej,day,corp)
        i = convDejDayCorpToFlatten(setup_docs,dej,day,corp)
        assert dej == convFlattenToDejDayCorp(setup_docs,i,days)[0]

        assert day == convFlattenToDejDayCorp(setup_docs,i,days)[1]
        assert corp == convFlattenToDejDayCorp(setup_docs,i, days)[2]
        cnt -= 1

def test_DSP_getDocShifts(setup_docs):
    init = getInitSchedule(setup_docs)
  #  pdb.set_trace()
    setup_docs.printScheduleInfo(init)
    """
    names = ['Raj', 'Shivam', 'Shreeya', 'Kartik']
    marks = [7, 9, 8, 5]
    div = ['A', 'A', 'C', 'B']
    id = [21, 52, 27, 38]
    print()
    print(f"{'Name' : <10}{'Marks' : ^10}{'Division' : ^10}{'ID' : >5}")

    for i in range(0,4):
        print(f"{names[i] : <10}{marks[i] : ^10}{div[i] : ^10}{id[i] : >5}") 

    """






