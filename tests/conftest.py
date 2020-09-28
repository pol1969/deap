
import pytest
from doc import DocSchedulingProblem

@pytest.fixture()
def setup_docs():
    print("setup_docs")  
    docs = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
    docShed = DocSchedulingProblem(10,docs)
    return docs,docShed
 
