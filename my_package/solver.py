"""Define a solver that advances the solution to a PDE in time, using some scheme."""

from my_package.function import MeshFunction
from my_package.scheme import ImplicitEuler, ExplicitEuler

import numpy as np

_SCHEME_INDEX = {
    "implicit_euler": ImplicitEuler,
    "explicit_euler": ExplicitEuler,
}


class Solver:
    """
    Time-stepping solver.

    The solver performs the tasks:
      - Initialization of the solution MeshFunction
      - Selection of a stepping algorithm
      - Advancing the solution to a specified final time, either returning the
        end state or the complete time-history of intermediate states.

    Parameters
    ----------
    mesh : Mesh
        Finite difference mesh for spatial discretization.
    operator : callable
        Operator acting on the solution MeshFunction.
    scheme : str
        Time-integration scheme key. Supports schemes in _SCHEME_INDEX.
    bc : BoundaryCondition
        Boundary condition handler(s).
    ic : callable or sequence of callables
        Initial condition function(s).
    initial_time : float
        Starting time of the simulation.
    dt : float
        Time-step size.
    n_solutions : int
        Number of solution fields (1 or 2 currently supported).
    """

    def __init__(self, mesh, operator, scheme, bc, ic, initial_time, dt, n_solutions):
        if scheme not in _SCHEME_INDEX:
            raise ValueError(f"Unknown scheme '{scheme}'.  Choose one of {list(_SCHEME_INDEX)}")
        if dt <= 0:
            raise ValueError("dt must be a positive number")
        if initial_time < 0:
            raise TypeError("initial_time must be a non-negative number")
        if n_solutions not in (1, 2):
            raise ValueError("n_solutions must be 1 or 2")

        self.mesh, self.operator, self.scheme, self.bc, self.ic = mesh, operator, scheme, bc, ic
        self.initial_time, self.dt = initial_time, dt

        if n_solutions == 1:
            self.solution = MeshFunction(self.mesh, func=ic)
            self.bc.apply(self.solution, self.initial_time)
        elif n_solutions == 2:
            self.solution = [MeshFunction(self.mesh, func=ic[0]), MeshFunction(self.mesh, func=ic[1])]
            self.bc[0].apply(self.solution, self.initial_time)
            self.bc[1].apply(self.solution, self.initial_time)

        stepper_class = _SCHEME_INDEX[scheme]
        self.stepper = stepper_class(self.solution, self.operator, self.bc, self.initial_time, n_solutions)

    def solve(self, final_time):
        """Advance the solution from initial_time to final_time using the selected scheme."""
        if final_time <= self.initial_time:
            raise ValueError("final_time must be > initial_time")
        if self.dt > (final_time - self.initial_time):
            raise ValueError("dt is larger than total integration time")

        n_steps = int(final_time / self.dt)

        if n_steps < 1:
            raise ValueError("No time steps to perform: check `dt` and `final_time`")

        for _ in range(n_steps):
            self.solution = self.stepper.step(self.dt)

        return self.solution

    def solve_history(self, final_time):
        """Advance the solution from `initial_time` to `final_time` using the configured scheme. Records all intermediate solutions."""
        if final_time <= self.initial_time:
            raise ValueError("final_time must be > initial_time")
        if self.dt > (final_time - self.initial_time):
            raise ValueError("dt is larger than total integration time")

        n_steps = int(final_time / self.dt)

        if n_steps < 1:
            raise ValueError("No time steps to perform: check `dt` and `final_time`")

        n_steps = int(final_time / self.dt)

        times = np.empty(n_steps)
        solutions = np.empty(n_steps, dtype=object)

        for n_step in range(n_steps):
            self.solution = self.stepper.step(self.dt)
            times[n_step] = self.dt * (n_step + 1)
            solutions[n_step] = self.solution.values.copy()

        return times, solutions
