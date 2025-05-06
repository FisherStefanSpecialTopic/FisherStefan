# flake8: noqa

import pytest
import numpy as np
from my_package.scheme import ExplicitEuler, ImplicitEuler
from my_package.function import MeshFunction
from my_package.mesh import Mesh
from my_package.boundary import NoneBC

def test_explicit_euler():
    mesh = Mesh(0, 1, n=4)
    function = MeshFunction(mesh, func=lambda x: np.ones_like(x))
    scheme = ExplicitEuler(function, operator=lambda u: u*0, bc=NoneBC(), initial_time=0.0, n_solutions=1)
    sol = scheme.step(0.1)
    assert np.allclose(sol.values, 1)

def test_implicit_euler():
    mesh = Mesh(0, 1, n=4)
    function = MeshFunction(mesh, func=lambda x: np.ones_like(x))
    scheme = ImplicitEuler(function, operator=lambda u: u*0, bc=NoneBC(), initial_time=0.0, n_solutions=1)
    sol = scheme.step(0.1)
    assert np.allclose(sol.values, 1)
