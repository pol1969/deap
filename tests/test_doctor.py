import pytest
from doc import DocSchedulingProblem
import numpy as np
import pdb

       

def test_all(setup_docs):
    # create a problem instance:
    
    assert type(setup_docs[1])==DocSchedulingProblem

def test_random_solution(setup_docs):
    randomSolution = np.random.randint(2, size=len(setup_docs[1]))
 #   pdb.set_trace()
    print("Random Solution = ")
    print(randomSolution)
    print()
