# flake8: noqa

import pytest
import numpy as np
from my_package.boundary import DirichletBC, NeumannBC, NoneBC
from my_package.mesh import Mesh
from my_package.function import MeshFunction

def test_dirichlet_bc():
    mesh = Mesh(0, 1, n=3)
    function = MeshFunction(mesh, func=lambda x: x)
    bc = DirichletBC(5, -1)
    bc.apply(function, t=0)
    assert function.values[0] == 5
    assert function.values[-1] == -1

def test_neumann_bc():
    mesh = Mesh(0, 1, n=3)
    function = MeshFunction(mesh, func=lambda x: x)
    bc = NeumannBC(2, -2)
    orig = function.values.copy()
    bc.apply(function, t=0)
    dx = function.mesh.dx
    assert function.values[0]  == pytest.approx(orig[1]  - 2 * dx)
    assert function.values[-1] == pytest.approx(orig[-2] + 2 * dx * 1)

def test_none_bc():
    mesh = Mesh(0, 1, n=3)
    function = MeshFunction(mesh, func=lambda x: x)
    bc = NoneBC()
    # should run without error
    bc.apply(function, t=0)

