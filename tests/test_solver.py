# flake8: noqa

import pytest
import numpy as np
from my_package.solver import Solver
from my_package.mesh import Mesh
from my_package.function import MeshFunction
from my_package.operator import LogisticGrowth
from my_package.boundary import NoneBC

def ic(x):
    return 0.1 * np.ones_like(x)

def test_solve_and_solve_history():
    mesh = Mesh(0, 1, n=5)
    op = LogisticGrowth()
    solver = Solver(mesh, op, "explicit_euler", NoneBC(), ic, initial_time=0.0, dt=0.01, n_solutions=1)
    final = solver.solve(0.05)
    assert np.all(final.values > 0.1)
    times, sols = solver.solve_history(0.03)
    assert len(times) == 3
    assert len(sols)  == 3

def test_solver_invalid_args():
    mesh = Mesh(0, 1, n=5)
    with pytest.raises(ValueError):
        Solver(mesh, lambda u: u, "unknown", NoneBC(), ic, 0.0, 0.01, 1)
    with pytest.raises(ValueError):
        Solver(mesh, lambda u: u, "explicit_euler", NoneBC(), ic, 0.0, -0.01, 1)
    solver = Solver(mesh, lambda u: u, "explicit_euler", NoneBC(), ic, 0.0, 0.01, 1)
    with pytest.raises(ValueError):
        solver.solve(0.0)  
