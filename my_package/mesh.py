"""Build a finite difference mesh."""

import numpy as np


class Mesh:
    """
    Build a 1D mesh on [a, b] either by specifying the number of nodes (n) or the node spacing (dx).

    Parameters
    ----------
    a : float
        Left endpoint of the interval.
    b : float
        Right endpoint of the interval; must be > a.
    n : int, optional
        Number of nodes. Must be >= 2.
    dx : float, optional
        Node spacing. Must be > 0 and < (b-a).
    """

    def __init__(self, a, b, *, n=None, dx=None):
        if b <= a:
            raise ValueError(f"Upper bound b={b} must be greater than lower bound a={a}")
        if (n is None) and (dx is None):
            raise ValueError("Please provide a number of nodes or a node spacing")
        if (n is not None) and (dx is not None):
            raise ValueError("Please provide a number of nodes or a node spacing, not both")
        if n is not None and n < 2:
            raise ValueError("Please provide at least two nodes")
        if dx is not None and dx <= 0:
            raise ValueError("Node spacing must be positive")
        if dx is not None and dx > (b - a):
            raise ValueError("Node spacing cannot be larger than the whole interval")

        self.a, self.b = a, b

        if n is not None:
            self.n = n
            self.dx = (self.b - self.a) / (self.n - 1)
        if dx is not None:
            self.n = 1 + int(round((self.b - self.a) / dx))
            self.dx = (self.b - self.a) / (self.n - 1)

        self.x = np.linspace(self.a, self.b, self.n)
