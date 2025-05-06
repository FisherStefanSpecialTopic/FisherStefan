"""
Provide the MeshFunction class, whose objects approximate functions over a given mesh.

It contains:
- Construction from either a list of values or a function.
- Dunder methods to perform a wide range of arithmetic operations between functions
- A shifting operator, necessary to compute finite differences in operator.py.
- String representation of MeshFunction.
"""

import numpy as np
import numbers


class MeshFunction:
    """
    Approximation of a function over a mesh. Provides support for arithmetic operations.

    Parameters
    ----------
    mesh : Mesh
        Spatial discretization object
    values : array_like, optional
        Vector of values at each node. Must be compatible with the mesh size
    func : callable, optional
        Function that takes the mesh and constructs the values of the approximating function
    """

    def __init__(self, mesh, *, values=None, func=None):
        self.mesh = mesh

        if (values is not None) and (func is not None):
            raise ValueError("You need to provide a vector or a function, not both")

        if values is not None:
            if len(values) != self.mesh.n:
                raise ValueError("The initialization vector has incompatible dimensions with the mesh")
            self.values = values.copy()
        elif func is not None:
            self.values = func(self.mesh.x)
        else:
            self.values = np.zeros(self.mesh.n)

    def __str__(self):
        """Represent function as a string."""
        return np.array2string(self.values, precision=2, suppress_small=True)

    def shift(self, n):
        """Shift the values of the function by one position."""
        if n not in (-1, 0, 1):
            raise ValueError("Shift must be -1, 0, or 1")

        result = MeshFunction(self.mesh)

        if n == 1:
            result.values[1:-1] = self.values[2:]
        if n == 0:
            result.values[1:-1] = self.values[1:-1]
        if n == -1:
            result.values[1:-1] = self.values[0:-2]

        result.values[0], result.values[-1] = 0, 0

        return result

    def __add__(self, other):
        """Add two functions, or a function and a number."""
        if isinstance(other, numbers.Number):
            return MeshFunction(self.mesh, values=self.values + other)
        elif isinstance(other, MeshFunction):
            if other.mesh.n != self.mesh.n:
                raise ValueError("Incompatible mesh sizes")
            return MeshFunction(self.mesh, values=self.values + other.values)

    def __sub__(self, other):
        """Subtract two functions, or a function and a number."""
        if isinstance(other, numbers.Number):
            return MeshFunction(self.mesh, values=self.values - other)
        elif isinstance(other, MeshFunction):
            if other.mesh.n != self.mesh.n:
                raise ValueError("Incompatible mesh sizes")
            return MeshFunction(self.mesh, values=self.values - other.values)

    def __rsub__(self, other):
        """Enable subtraction when the left argument is a number."""
        if isinstance(other, numbers.Number):
            return MeshFunction(self.mesh, values=other - self.values)

    def __mul__(self, other):
        """Multiply two functions element by element, or a function with a number."""
        if isinstance(other, numbers.Number):
            return MeshFunction(self.mesh, values=self.values * other)
        elif isinstance(other, MeshFunction):
            if other.mesh.n != self.mesh.n:
                raise ValueError("Incompatible mesh sizes")
            return MeshFunction(self.mesh, values=self.values * other.values)

    def __rmul__(self, other):
        """Enable multiplication when the left argument is a number."""
        return self.__mul__(other)

    def __truediv__(self, scalar):
        """Divide a function by a number."""
        return MeshFunction(self.mesh, values=self.values / scalar)

    def __rtruediv__(self, scalar):
        """Divide a number by a function, which returns the reciprocal of the function."""
        return MeshFunction(self.mesh, values=scalar / self.values)

    def __pow__(self, scalar):
        """Take the nth power of a function."""
        if scalar > 0:
            return MeshFunction(self.mesh, values=self.values ** scalar)
        elif scalar < 0:
            return MeshFunction(self.mesh, values=1 / (self.values ** (-scalar)))
        else:
            return MeshFunction(self.mesh, func=lambda x: 1 + x * 0)
