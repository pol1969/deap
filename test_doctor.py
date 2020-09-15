from doc import NurseSchedulingProblem
import numpy as np

def test_all():
    # create a problem instance:
    
    docs = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
    docShed = NurseSchedulingProblem(10,docs)
    assert type(docShed)==NurseSchedulingProblem

    randomSolution = np.random.randint(2, size=len(docShed))
    print("Random Solution = ")
    print(randomSolution)
    print()

    docShed.printScheduleInfo(randomSolution)

    print("Total Cost = ", docShed.getCost(randomSolution))

