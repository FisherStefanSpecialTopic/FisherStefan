"""
Solve and visualize the Fisher-Kolmogorov reaction-diffusion equation.

Obtains and plots the exact solution.
Obtains and plots numerical approximations to the solution, using finite differences and explicit or implicit time stepping.
Defines a finite difference mesh, a spatial operator, boundary conditions, and an initial condition.
"""
from my_package.mesh import Mesh
from my_package.function import MeshFunction
from my_package.operator import SecondDD, LogisticGrowth
from my_package.boundary import NeumannBC
from my_package.solver import Solver
from my_package.graph import Graph

import numpy as np

lower_bound = 0
upper_bound = 40

mesh = Mesh(lower_bound, upper_bound, n=401)

operator = SecondDD() + LogisticGrowth()


def ic(x):
    """
    Define the initial condition for the Fisher-Kolmogorov problem.

    Composed of two branches:
    - A constant for x < 10
    - An exponentially decaying tail for x >= 10
    """
    branch_1 = 0.5
    branch_2 = 0.5 * np.exp((-2 * (x - 10)) / np.sqrt(6))

    return np.where(x < 10, branch_1, branch_2)


bc = NeumannBC(0, 0)

initial_time, initial_dt = 0, 0.001
final_time = 2 * np.sqrt(6)

print("Fisher Kolmogorov Equation - Plot Options")
print("1) Exact solution")
print("2) Snapshots of explicit numerical solution + Exact solution")
print("3) Snapshots of implicit numerical solution + Exact solution")
option = int(input("Please select a plot option: "))

if option == 1:
    c = 5 / np.sqrt(6)
    k = 1.2
    exact_mesh = Mesh(lower_bound, upper_bound, n=1000)
    exact_solution = MeshFunction(exact_mesh, func=lambda x: (1 + (np.sqrt(k) - 1) * np.exp((x - c * final_time) / np.sqrt(6))) ** (-2))
    graph = Graph(xlabel="x", ylabel="u(x)", title="Fisher-Kolmogorov Equation. Exact Solution", xlim=[lower_bound, upper_bound], ylim=[0, 1.2])
    graph.add_solution(exact_mesh, exact_solution, color="C4", linestyle="--", linewidth=2, label="Exact")
    graph.show()

elif option == 2:
    explicit_solver = Solver(mesh, operator, "explicit_euler", bc, ic, initial_time, initial_dt, 1)
    explicit_times, explicit_solutions = explicit_solver.solve_history(final_time)

    c = 5 / np.sqrt(6)
    x_center = c * final_time
    idx = np.abs(mesh.x - x_center).argmin()
    k = 1 / explicit_solutions[-1][idx]
    exact_mesh = Mesh(lower_bound, upper_bound, n=1000)
    exact_solution = MeshFunction(exact_mesh, func=lambda x: (1 + (np.sqrt(k) - 1) * np.exp((x - c * final_time) / np.sqrt(6))) ** (-2))

    graph = Graph(xlabel="x", ylabel="u(x)", title="Fisher-Kolmogorov Equation. Explicit Solver vs Exact Solution", xlim=[lower_bound, upper_bound], ylim=[0, 1.2])
    graph.add_solution(mesh, MeshFunction(mesh, func=ic), color="C1", linestyle="-", linewidth=2, label="t = 0")
    graph.add_solution(mesh, MeshFunction(mesh, values=explicit_solutions[-3899]), color="C2", linestyle="-", linewidth=2, label="t = 1")
    graph.add_solution(mesh, MeshFunction(mesh, values=explicit_solutions[-1]), color="C3", linestyle="-", linewidth=2, label=r"t = $2\sqrt{6}$")
    graph.add_solution(exact_mesh, exact_solution, color="C4", linestyle="--", linewidth=2, label="Exact")
    graph.show()

elif option == 3:
    implicit_solver = Solver(mesh, operator, "implicit_euler", bc, ic, initial_time, initial_dt, 1)
    implicit_times, implicit_solutions = implicit_solver.solve_history(final_time)

    c = 5 / np.sqrt(6)
    x_center = c * final_time
    idx = np.abs(mesh.x - x_center).argmin()
    k = 1 / implicit_solutions[-1][idx]
    exact_mesh = Mesh(lower_bound, upper_bound, n=1000)
    exact_solution = MeshFunction(exact_mesh, func=lambda x: (1 + (np.sqrt(k) - 1) * np.exp((x - c * final_time) / np.sqrt(6))) ** (-2))

    graph = Graph(xlabel="x", ylabel="u(x)", title="Fisher-Kolmogorov Equation. Implicit Solver vs Exact Solution", xlim=[lower_bound, upper_bound], ylim=[0, 1.2])
    graph.add_solution(mesh, MeshFunction(mesh, func=ic), color="C1", linestyle="-", linewidth=2, label="t = 0")
    graph.add_solution(mesh, MeshFunction(mesh, values=implicit_solutions[-3899]), color="C2", linestyle="-", linewidth=2, label="t = 1")
    graph.add_solution(mesh, MeshFunction(mesh, values=implicit_solutions[-1]), color="C3", linestyle="-", linewidth=2, label=r"t = $2\sqrt{6}$")
    graph.add_solution(exact_mesh, exact_solution, color="C4", linestyle="--", linewidth=2, label="Exact")
    graph.show()

else:
    raise ValueError("Not a valid plot option")
