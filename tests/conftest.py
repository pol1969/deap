
import pytest
from doc import DocSchedulingProblem
from doc10 import Doc10SchedulingProblem
import pandas as pd
import datetime as dt

@pytest.fixture()
def setup_docs():
#    print("setup_docs")  
    p = pd.read_csv("lk_1.csv")

    month = dt.datetime.now().month +1
    year = dt.datetime.now().year
    if month==13:
        month =1
        year = year +1

    docShed = DocSchedulingProblem(10,p,month,year)
    return docShed

@pytest.fixture()
def setup10_docs():
#    print("setup_docs")  
    p = pd.read_csv("lk_1.csv")
    
    month = dt.datetime.now().month +1
    year = dt.datetime.now().year
    if month==13:
        month =1
        year = year +1


    docShed = Doc10SchedulingProblem(10,p,month,year)

    return docShed
 
