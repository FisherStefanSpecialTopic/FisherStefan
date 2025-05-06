# flake8: noqa

import pytest
import numpy as np
from my_package.mesh import Mesh

def test_n_parameters():
    with pytest.raises(ValueError):
        Mesh(0, 1)                
    with pytest.raises(ValueError):
        Mesh(0, 1, n=10, dx=0.1)  

def test_parameter_relation():
    n, expected_dx = 5, 0.25
    m = Mesh(0, 1, n=n)
    assert m.dx == expected_dx

def test_parameter_relation_2():
    m = Mesh(0, 2, dx=0.5)
    assert m.n == 5                   
    assert m.dx == pytest.approx(0.5)
    assert np.allclose(m.x, np.linspace(0, 2, 5))

def test_invalid_bounds_and_types():
    with pytest.raises(ValueError):
        Mesh(1, 1, n=2)                 
    with pytest.raises(ValueError):
        Mesh(2, 1, n=2)                 
    with pytest.raises(TypeError):
        Mesh("a", 1, n=2)
    with pytest.raises(TypeError):
        Mesh(0, "b", dx=0.1)
    with pytest.raises(TypeError):
        Mesh(0, 1, n=2.5)
    with pytest.raises(ValueError):
        Mesh(0, 1, dx=-0.1)