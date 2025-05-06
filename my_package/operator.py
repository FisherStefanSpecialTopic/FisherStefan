"""
Apply a range of operators to a function (or pair of functions).

Combine the operators through different operations, to build more complex operators.
"""

from my_package.function import MeshFunction


class Operator:
    """
    Complex operator that combines smaller operators.

    It takes a left and right term (either complex operators such as this one, or simpler ones), and combines them using the operation
    """

    def __init__(self, left, right, op):
        self.left, self.right, self.op = left, right, op

    def __call__(self, function):
        """
        Apply the complex operator to a function, by applying the smaller operators of which it is composed.

        The case of the operation @ applies the operator to two functions. It applies the left term to the first function,
        and the right term to the second function. It then multiplies them.
        """
        if self.op == "+":
            return self.left(function) + self.right(function)
        elif self.op == "-":
            return self.left(function) - self.right(function)
        elif self.op == "*":
            return self.left(function) * self.right(function)
        elif self.op == "@":
            return self.left(function[0]) * self.right(function[1])
        else:
            raise ValueError(f"Unsupported operation {self.op}")

    def __add__(self, term):
        """Add an operator to a term."""
        return Operator(self, term, "+")

    def __sub__(self, term):
        """Subtract a term to a operator."""
        return Operator(self, term, "-")

    def __mul__(self, term):
        """Multiply a term to a operator."""
        return Operator(self, term, "*")

    def __matmul__(self, term):
        """Multiply an operator applied to a function, to a term applied to another function."""
        return Operator(self, term, "@")


class Term:
    """Simple operator applied to a function."""

    def __add__(self, other):
        """Add two terms."""
        return Operator(self, other, "+")

    def __sub__(self, other):
        """Subtract two terms."""
        return Operator(self, other, "-")

    def __mul__(self, other):
        """Multiply two terms."""
        return Operator(self, other, "*")

    def __matmul__(self, other):
        """Multiply a term applied to a function, with another term applied to another function."""
        return Operator(self, other, "@")


class FirstDD(Term):
    """First divided difference."""

    def __init__(self, version):
        super().__init__()
        self.version = version

    def __call__(self, function):
        """
        Apply a first divided difference to a function.

        It can be a central, backward or forward divided difference.

        The special case is reserved for the Stefan condition of the Fisher-Stefan equation
        """
        if self.version == "central":
            return (function.shift(1) - function.shift(-1)) / (2 * function.mesh.dx)
        elif self.version == "backward":
            return (function.shift(0) - function.shift(-1)) / (function.mesh.dx)
        elif self.version == "forward":
            return (function.shift(1) - function.shift(0)) / (function.mesh.dx)
        elif self.version == "special":
            # Hard-coded parameter kappa, see essay
            return (-0.906610965581149 * function.values[-2]) / function.mesh.dx
        else:
            raise ValueError("Not a valid spatial divided difference version")


class SecondDD(Term):
    """Symmetric second divided difference."""

    def __init__(self):
        super().__init__()

    def __call__(self, function):
        """Apply the second divided difference to a function."""
        return (function.shift(1) - 2 * function.shift(0) + function.shift(-1)) / (function.mesh.dx ** 2)


class LogisticGrowth(Term):
    """Logistic growth term."""

    def __init__(self):
        super().__init__()

    def __call__(self, function):
        """Apply the logistic growth operator to a function."""
        return function * (1 - function)


class Power(Term):
    """Take nth power of a function."""

    def __init__(self, power):
        super().__init__()
        self.power = power

    def __call__(self, function):
        """Apply the power operator to a function."""
        return function ** self.power


class Extract(Term):
    """Extract the argument from a function."""

    def __init__(self):
        super().__init__()

    def __call__(self, function):
        """Apply the extraction operator to a function."""
        return MeshFunction(function.mesh, values=function.mesh.x)
