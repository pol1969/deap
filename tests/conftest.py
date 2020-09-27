
import pytest
from doc import NurseSchedulingProblem

@pytest.fixture()
def setup_docs():
    print("setup_docs")  
    docs = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
    docShed = NurseSchedulingProblem(10,docs)
    return docs,docShed
 
