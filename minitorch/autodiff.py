from dataclasses import dataclass
from collections import defaultdict
from typing import Any, Iterable, List, Tuple

from typing_extensions import Protocol

# ## Task 1.1
# Central Difference calculation


def central_difference(f: Any, *vals: Any, arg: int = 0, epsilon: float = 1e-6) -> Any:
    r"""
    Computes an approximation to the derivative of `f` with respect to one arg.

    See :doc:`derivative` or https://en.wikipedia.org/wiki/Finite_difference for more details.

    Args:
        f : arbitrary function from n-scalar args to one value
        *vals : n-float values $x_0 \ldots x_{n-1}$
        arg : the number $i$ of the arg to compute the derivative
        epsilon : a small constant

    Returns:
        An approximation of $f'_i(x_0, \ldots, x_{n-1})$
    """

    lhs = vals[:arg] + (vals[arg] + epsilon,) + vals[arg + 1:]
    rhs = vals[:arg] + (vals[arg] - epsilon,) + vals[arg + 1:]
    return (f(*lhs) - f(*rhs)) / (2 * epsilon)


variable_count = 1


class Variable(Protocol):
    def accumulate_derivative(self, x: Any) -> None:
        pass

    @property
    def unique_id(self) -> int:
        pass

    def is_leaf(self) -> bool:
        pass

    def is_constant(self) -> bool:
        pass

    @property
    def parents(self) -> Iterable["Variable"]:
        pass

    def chain_rule(self, d_output: Any) -> Iterable[Tuple["Variable", Any]]:
        pass


def topological_sort(variable: Variable) -> Iterable[Variable]:
    """
    Computes the topological order of the computation graph.

    Args:
        variable: The right-most variable

    Returns:
        Non-constant Variables in topological order starting from the right.
    """
    def visit(variable: Variable, color: dict[int, int], result: list[Variable]) -> None:
        if variable.unique_id in color and color[variable.unique_id] == 2:
            return
        if variable.unique_id in color and color[variable.unique_id] == 1:
            raise ValueError("Graph has at least one cycle")
        color[variable.unique_id] = 1

        for parent in variable.parents:
            visit(parent, color, result)
        color[variable.unique_id] = 2
        result.append(variable)

    result: list[Variable] = []
    color: dict[int, int] = {}

    visit(variable, color, result)

    return result


def backpropagate(variable: Variable, deriv: Any) -> None:
    """
    Runs backpropagation on the computation graph in order to
    compute derivatives for the leave nodes.

    Args:
        variable: The right-most variable
        deriv  : Its derivative that we want to propagate backward to the leaves.

    No return. Should write to its results to the derivative values of each leaf through `accumulate_derivative`.
    """
    queue = list(topological_sort(variable))

    grads = defaultdict(lambda: 0)

    grads[variable.unique_id] = deriv

    for var in reversed(queue):
        if var.is_leaf():
            var.accumulate_derivative(grads[var.unique_id])
        else:
            for scalar, devivative in var.chain_rule(grads[var.unique_id]):
                grads[scalar.unique_id] += devivative
                    


@dataclass
class Context:
    """
    Context class is used by `Function` to store information during the forward pass.
    """

    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:
        "Store the given `values` if they need to be used during backpropagation."
        if self.no_grad:
            return
        self.saved_values = values

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        return self.saved_values
