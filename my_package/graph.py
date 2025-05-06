"""Construct a graph object to represent functions in their meshes."""
import matplotlib.pyplot as plt


class Graph:
    """
    Graph that can hold multiple functions.

    To initialize it, it is possible (though not necessary) to set a number of plotting parameters
    """

    def __init__(self, *, xlabel=None, ylabel=None, title=None, xlim=None, ylim=None):
        self.fig, self.ax = plt.subplots()
        self.ax.set_xlabel(xlabel, fontsize=18, fontweight='bold')
        self.ax.set_ylabel(ylabel, fontsize=18, fontweight='bold')
        self.ax.set_title(title, fontsize=20, fontweight='bold')

        self.ax.set_xlim(xlim[0], xlim[1])
        self.ax.set_ylim(ylim[0], ylim[1])

        self.ax.tick_params(axis='both', which='major', labelsize=15)

    def add_solution(self, mesh, solution, *, color=None, linestyle="-", linewidth=2, label=None):
        """Add solution to graph."""
        self.ax.plot(mesh.x, solution.values, color=color, linestyle=linestyle, linewidth=linewidth, label=label)
        self.ax.legend(fontsize=20)

    def show(self):
        """Visualize graph."""
        plt.show()
