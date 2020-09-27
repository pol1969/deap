import pytest
from doc import NurseSchedulingProblem
import numpy as np
import pdb

@pytest.fixture()
def setup_docs():
    print("setup_docs")  
    docs = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
    docShed = NurseSchedulingProblem(10,docs)
    return docs,docShed
        

def test_all(setup_docs):
    # create a problem instance:
    
    assert type(setup_docs[1])==NurseSchedulingProblem

def test_random_solution(setup_docs):
    randomSolution = np.random.randint(2, size=len(setup_docs[1]))
    pdb.set_trace()
    print("Random Solution = ")
    print(randomSolution)
    print()
