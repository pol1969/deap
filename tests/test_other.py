import numpy as np
from bool_gen import binary_mask_random 
import pytest

@pytest.mark.skip()
def test_random_array():
    ar = np.random.randint(2,size=10)
    s = binary_mask_random(3,4,10)
    print(s)
    assert 1 == 1

