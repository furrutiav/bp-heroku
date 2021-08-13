"""
Uso del solver.
"""
from bongard_problem import BongardProblem
from solver import BPSolver


def solve(selection=0, n_attributes=0):
    filename = 'some_image.jpg'
    bp = BongardProblem(filename)
    solver = Solver(bp, n_select=selection,
                    n_lasso=n_attributes, alpha_lasso=0.1)
    solver.default_solve()
    output = {}
    output["models"] = solver.solutions
    output["solution"] = solver.atts
    return output


if __name__ == "__main__":
    print(solve())
