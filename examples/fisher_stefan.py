"""
Solve and visualize the Fisher-Stefan reaction-diffusion equation.

Obtains and plots the exact solution.
Obtains and plots numerical approximations to the solution, using finite differences and explicit time stepping.
Defines a finite difference mesh, a spatial operator, boundary conditions, and an initial condition.
"""
from my_package.mesh import Mesh
from my_package.function import MeshFunction
from my_package.operator import FirstDD, SecondDD, LogisticGrowth, Extract, Power
from my_package.boundary import FisherStefanBC, NoneBC
from my_package.solver import Solver
from my_package.graph import Graph

import numpy as np

from my_package.weierstrass import WeierstrassP


def weier(z, g2, g3):
    """Return the real value of the WeierstrassP function at z, with invariants g2 and g3."""
    return WeierstrassP(z, g2, g3).real


def u(z, k, g3):
    """Return the exact solution of the Fisher-Stefan equation at z (given c = -5/sqrt(6)), with parameters k and g3."""
    arg = np.exp(z / np.sqrt(6)) - k
    return np.exp((2 * z) / np.sqrt(6)) * weier(arg, 0, g3)


mesh = Mesh(0, 1, n=101)

operator0 = (SecondDD() @ Power(-2)) + ((Extract() * FirstDD("special") * FirstDD("central")) @ Power(-2)) + (LogisticGrowth() @ Power(0))
operator1 = FirstDD("special") @ Power(-1)
operator = [operator0, operator1]

ic0 = lambda x: 0.5 + x * 0
ic1 = lambda x: 100 + x * 0
ic = [ic0, ic1]

bc0 = FisherStefanBC(0, 0)
bc1 = NoneBC()
bc = [bc0, bc1]

initial_time, initial_dt = 0, 0.001
final_time = 10 * np.sqrt(6)

print("Fisher-Kolmogorov Equation - Plot Options")
print("1) Exact solution")
print("2) Snapshots of explicit numerical solution + Exact solution")
option = int(input("Please select a plot option: "))

if option == 1:
    # Hard-coded parameters k and g3, see essay
    k = -3.3916324186463555218
    g3 = -14.56

    zv = np.linspace(-30, 0.5, num=1000)
    values = np.array([u(z, k, g3) for z in zv])

    exact_mesh = Mesh(-30 + 85, 0.5 + 85, n=1000)
    exact_solution = MeshFunction(exact_mesh, values=values)

    graph = Graph(xlabel="x", ylabel="u(x)", title="Fisher-Stefan Equation. Exact Solution", xlim=[60, 100], ylim=[0, 1.2])
    graph.add_solution(exact_mesh, exact_solution, color="C2", linestyle="--", linewidth=2, label="Exact Solution")
    graph.show()

elif option == 2:
    graph = Graph(xlabel="x", ylabel="u(x)", title="Fisher-Stefan Equation. Explicit Solver vs Exact Solution", xlim=[60, 100], ylim=[0, 1.2])

    explicit_solver = Solver(mesh, operator, "explicit_euler", bc, ic, initial_time, initial_dt, 2)

    explicit_solution = explicit_solver.solve(final_time / 10)
    mesh = Mesh(0, explicit_solution[1].values[0], n=101)
    graph.add_solution(mesh, explicit_solution[0], color="C0", linestyle="-", linewidth=2, label=r"t = $\sqrt{6}$")

    explicit_solution = explicit_solver.solve(final_time / 2)
    mesh = Mesh(0, explicit_solution[1].values[0], n=101)
    graph.add_solution(mesh, explicit_solution[0], color="C1", linestyle="-", linewidth=2, label=r"t = $5\sqrt{6}$")

    explicit_solution = explicit_solver.solve(final_time)
    idx = np.abs(explicit_solution[0].values - 0.5).argmin()
    mesh = Mesh(0, explicit_solution[1].values[0], n=101)
    s = mesh.x[idx]
    graph.add_solution(mesh, explicit_solution[0], color="C2", linestyle="-", linewidth=2, label=r"t = $10\sqrt{6}$")

    k = -3.3916324186463555218
    g3 = -14.56
    zv = np.linspace(-30, 0.5, num=1000)
    values = np.array([u(z, k, g3) for z in zv])
    exact_mesh = Mesh(-30 + s, 0.5 + s, n=1000)
    exact_solution = MeshFunction(exact_mesh, values=values)

    graph.add_solution(exact_mesh, exact_solution, color="C3", linestyle="--", linewidth=2, label="Exact")
    graph.show()
else:
    raise ValueError("Not a valid plot option")
