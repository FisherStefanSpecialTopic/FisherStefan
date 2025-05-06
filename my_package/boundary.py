"""
Define boundary conditions on one-dimensional meshes.

Includes
- Dirichlet and Neumann BCs on both sides
- Ad-hoc BCs for the Fisher-Stefan problem
- A class to apply BCs that have no effect, for functions with no spatial dependency
"""
import numbers


class BoundaryCondition():
    """
    Abstract base for time dependent boundary conditions.

    Parameters
    ----------
    left : float or callable
        Left boundary value or time-dependent function.
    right : float or callable
        Right boundary value or time-dependent function.
    """

    def __init__(self, left, right):
        if isinstance(left, numbers.Number):
            self.left = lambda t: left
        elif callable(left):
            self.left = left

        if isinstance(right, numbers.Number):
            self.right = lambda t: right
        elif callable(right):
            self.right = right

    def apply(self, function, t):
        """Abstract method that serves as template for children classes."""
        raise NotImplementedError("Abstract Method")


class DirichletBC(BoundaryCondition):
    """Dirichlet boundary conditions, which prescribe the end values of a MeshFunction as the evaluation of a time-dependent function."""

    def __init__(self, left, right):
        super().__init__(left, right)

    def apply(self, function, t):
        """Apply the Dirichlet BCs to a specific function."""
        function.values[0] = self.left(t)
        function.values[-1] = self.right(t)


class NeumannBC(BoundaryCondition):
    """
    Neumann boundary conditions.

    They establish a relationship between the last and penultimate values of a MeshFunction
    """

    def __init__(self, left, right):
        super().__init__(left, right)

    def apply(self, function, t):
        """Apply the Neumann BCs to a specific function."""
        function.values[0] = function.values[1] - function.mesh.dx * self.left(t)
        function.values[-1] = function.values[-2] - function.mesh.dx * self.right(t)


class FisherStefanBC(BoundaryCondition):
    """
    Ad-hoc boundary conditions for the Fisher-Stefan problem.

    They consist of a Dirichlet BC on the right and a Neumann BC on the left.
    """

    def __init__(self, left, right):
        super().__init__(left, right)

    def apply(self, function, t):
        """Apply the Fisher-Stefan BCs to a specific function."""
        function[0].values[0] = function[0].values[1] - function[0].mesh.dx * self.left(t)
        function[0].values[-1] = self.right(t)


class NoneBC():
    """Blank boundary conditions that do not change the function they are applied to."""

    def apply(self, function, t):
        """Apply the null BCs to a specific function."""
        pass
