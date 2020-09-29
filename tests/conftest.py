
import pytest
from doc import DocSchedulingProblem
import pandas as pd
import datetime as dt

@pytest.fixture()
def setup_docs():
    print("setup_docs")  
    p = pd.read_csv("lk_1.csv")
    docShed = DocSchedulingProblem(10,p,dt.datetime.now().month+1,dt.datetime.now().year)
    return docShed
 
