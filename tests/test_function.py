# flake8: noqa

import pytest
import numpy as np
from my_package.function import MeshFunction
from my_package.mesh import Mesh


def test_constructor():
    mesh = Mesh(0, 1, n=5)
    vals = [1, 2, 3, 4, 5]
    mf = MeshFunction(mesh, values=vals)
    assert np.allclose(mf.values, vals)

def test_constructor2():
    mesh = Mesh(0, 1, n=5)
    mf = MeshFunction(mesh, func=lambda x: x ** 2)
    assert np.allclose(mf.values, mesh.x ** 2)

def test_n_parameters():
    mesh = Mesh(0, 1, n=5)
    with pytest.raises(ValueError):
        MeshFunction(mesh, values=[1] * 5, func=lambda x: x)

def test_n_parameters_2():
    mesh = Mesh(0, 1, n=5)
    mf = MeshFunction(mesh)
    assert np.all(mf.values == 0)

def test_shift():
    mesh = Mesh(0, 1, n=5)
    vals = np.arange(5, dtype=float)
    mf = MeshFunction(mesh, values=vals)
    shifted = mf.shift(1)
    assert shifted.values[1:-1].tolist() == vals[2:].tolist()
    with pytest.raises(ValueError):
        wrong_shifted = mf.shift(2)  

def test_arithmetic():
    mesh = Mesh(0, 1, n=5)
    mf = MeshFunction(mesh, func=lambda x: x)
    other = MeshFunction(mesh, func=lambda x: 2 * x)
    assert np.allclose((mf + other).values, 3 * mesh.x)
    assert np.allclose((mf * 2).values, 2 * mesh.x)
    assert np.allclose((2 * mf).values, 2 * mesh.x)

def test_arithmetic_2():
    mesh = Mesh(0, 1, n=5)
    mf = MeshFunction(mesh, func=lambda x: x + 1)
    sq = mf ** 2
    assert np.allclose(sq.values, (mesh.x + 1)**2)
    with pytest.raises(TypeError):
        wrong_value = mf ** "foo"
