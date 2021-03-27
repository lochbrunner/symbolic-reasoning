import sympy as sp
import logging
from common.creator_utils import create_term
from common.solving_problems import create_solving_problems

logger = logging.getLogger(__name__)


def solution_filter(solution):
    print(solution)
    return len(solution) == 2


def create_quadratic_equations(**kwargs):
    x = sp.symbols('x')
    return create_solving_problems(
        operations_reservoir=('-', '+', '*', '/'),
        additional_reservoir=(sp.I, x ** 2),
        desc='quadaric',
        solution_filter=lambda solution: len(solution) == 2,
        **kwargs
    )


def create_exponential(**kwargs):
    return create_solving_problems(
        operations_reservoir=('-', '+', '*', '/', '^'),
        additional_reservoir=(sp.I,),
        desc='exponential',
        **kwargs
    )
