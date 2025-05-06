# flake8: noqa

import pytest
import numpy as np
from my_package.operator import FirstDD, SecondDD, LogisticGrowth, Power, Extract, Operator
from my_package.function import MeshFunction
from my_package.mesh import Mesh

def test_firstdd_central():
    mesh = Mesh(0, 1, n=7)
    function = MeshFunction(mesh, func=lambda x: x)
    deriv = FirstDD("central")(function)
    assert np.allclose(deriv.values[1:-1], 1)
    with pytest.raises(ValueError):
        FirstDD("invalid")(function)

def test_seconddd():
    mesh = Mesh(0, 1, n=7)
    function = MeshFunction(mesh, func=lambda x: x)
    sec = SecondDD()(function)
    assert np.allclose(sec.values, 0)

def test_logistic_growth():
    mesh = Mesh(0, 1, n=7)
    function = MeshFunction(mesh, func=lambda x: x)
    lg = LogisticGrowth()(function)
    assert np.allclose(lg.values, mesh.x * (1 - mesh.x))

def test_power_term():
    mesh = Mesh(0, 1, n=7)
    function = MeshFunction(mesh, func=lambda x: x)
    p3 = Power(3)(function)
    assert np.allclose(p3.values, mesh.x**3)

def test_extract():
    mesh = Mesh(0, 1, n=7)
    function = MeshFunction(mesh, func=lambda x: x)
    ext = Extract()(function)
    assert np.allclose(ext.values, mesh.x)

def test_operator_composition():
    mesh = Mesh(0, 1, n=7)
    function = MeshFunction(mesh, func=lambda x: x)
    op = SecondDD() + LogisticGrowth()
    res = op(function)
    assert np.allclose(res.values, mesh.x * (1 - mesh.x))
