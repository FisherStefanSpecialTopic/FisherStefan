"""Define time-advancement schemes to be called from the solver."""

from my_package.function import MeshFunction

from scipy.sparse.linalg import LinearOperator, gmres


class Scheme:
    """
    Base class for time-advancement schemes.

    Attributes
    ----------
    solution : MeshFunction or list of MeshFunction
        Current solution(s).
    operator : callable or sequence of callables
        Spatial operator(s) to be applied to each solution
    bc : BoundaryCondition
        Boundary condition handler(s).
    time : float
        Current simulation time.
    n_solutions : int
        Number of solutions (1 or 2 currently supported).
    """

    def __init__(self, solution, operator, bc, initial_time, n_solutions):
        self.solution, self.operator, self.bc, self.time, self.n_solutions = solution, operator, bc, initial_time, n_solutions


class ExplicitEuler(Scheme):
    """
    Explicit Euler time stepping scheme.

    Updates the solution via:
        u_{n+1} = u_n + dt * operator(u_n),
    and later enforces boundary conditions.
    """

    def __init__(self, solution, operator, bc, initial_time, n_solutions):
        super().__init__(solution, operator, bc, initial_time, n_solutions)

    def step(self, dt):
        """Perform one time step using the explicit Euler method."""
        self.time += dt

        if self.n_solutions == 1:
            self.solution = self.solution + dt * self.operator(self.solution)
            self.bc.apply(self.solution, self.time)
        elif self.n_solutions == 2:
            temp_1 = self.solution[1] + dt * self.operator[1](self.solution)
            self.bc[1].apply(self.solution, self.time)
            temp_0 = self.solution[0] + dt * self.operator[0](self.solution)
            self.bc[0].apply(self.solution, self.time)
            self.solution[1] = temp_1
            self.solution[0] = temp_0

        return self.solution


class ImplicitEuler(Scheme):
    """
    Implicit Euler time stepping scheme.

    Updates the solution by solving the system:
        (I - dt·operator(u_n)) u_{n+1} = u_n,
    through GMRES. It also enforces boundary conditions.
    """

    def __init__(self, solution, operator, bc, initial_time, n_solutions):
        if n_solutions == 2:
            raise ValueError("Implicit Euler not currently supported for two approximants")
        super().__init__(solution, operator, bc, initial_time, n_solutions)

    def matvec(self, v):
        """Auxiliary function needed to run GMRES."""
        temp = MeshFunction(self.solution.mesh, values=v)
        self.bc.apply(temp, self.time)

        result = temp - self.dt * self.operator(temp)

        return result.values

    def step(self, dt):
        """Perform one time step using the implicit Euler method."""
        self.dt = dt
        self.time += self.dt

        self.bc.apply(self.solution, self.time)

        a = LinearOperator((self.solution.mesh.n, self.solution.mesh.n), self.matvec, dtype=self.solution.values.dtype)

        b = MeshFunction(self.solution.mesh, values=self.solution.values)

        self.solution.values, info = gmres(a, b.values, rtol=1e-4)

        if info != 0:
            raise RuntimeError(f"GMRES failed to converge (info={info})")

        self.bc.apply(self.solution, self.time)

        return self.solution
