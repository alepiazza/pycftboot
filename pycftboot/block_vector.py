# Regular sympy is slow but we only use it for quick access to Gegenbauer polynomials
# Even this could be removed since our conformal block code is needlessly general
from symengine.lib.symengine_wrapper import (
    Symbol, Integer, DenseMatrix, factorial, sqrt
)
import sympy

from .common import rf, delta_pole, unitarity_bound
from .constants import r_cross, one, two, cutoff, tiny, delta


class LeadingBlockVector:
    def __init__(self, dim, l, m_max, n_max, delta_12, delta_34):
        self.spin = l
        self.m_max = m_max
        self.n_max = n_max
        self.chunks = []

        r = Symbol('r')
        eta = Symbol('eta')
        nu = (dim / Integer(2)) - 1
        derivative_order = m_max + 2 * n_max

        # With only a derivatives, we never need eta derivatives
        off_diag_order = derivative_order
        if n_max == 0:
            off_diag_order = 0

        # We cache derivatives as we go
        # This is because csympy can only compute them one at a time, but it's faster anyway
        old_expression = self.leading_block(nu, r, eta, l, delta_12, delta_34)

        for n in range(0, off_diag_order + 1):
            chunk = []
            for m in range(0, derivative_order - n + 1):
                if n == 0 and m == 0:
                    expression = old_expression
                elif m == 0:
                    old_expression = old_expression.diff(eta)
                    expression = old_expression
                else:
                    expression = expression.diff(r)

                chunk.append(expression.subs({r: r_cross, eta: 1}))
            self.chunks.append(DenseMatrix(len(chunk), 1, chunk))

    def leading_block(self, nu, r, eta, l, delta_12, delta_34):
        if self.n_max == 0:
            ret = 1
        elif nu == 0:
            ret = sympy.chebyshevt(l, eta)
        else:
            ret = factorial(l) * sympy.gegenbauer(l, nu, eta) / rf(2 * nu, l)

        # Time saving special case
        if delta_12 == delta_34:
            return ((-1) ** l) * ret / (((1 - r ** 2) ** nu) * sqrt((1 + r ** 2) ** 2 - 4 * (r * eta) ** 2))
        else:
            return ((-1) ** l) * ret / (((1 - r ** 2) ** nu) * ((1 + r ** 2 + 2 * r * eta) ** ((one + delta_12 - delta_34) / two)) * ((1 + r ** 2 - 2 * r * eta) ** ((one - delta_12 + delta_34) / two)))


class MeromorphicBlockVector:
    def __init__(self, leading_block):
        # A chunk is a set of r derivatives for one eta derivative
        # The matrix that should multiply a chunk is just R restricted to the right length
        self.chunks = []

        for j in range(0, len(leading_block.chunks)):
            rows = leading_block.chunks[j].nrows()
            self.chunks.append(DenseMatrix(rows, 1, [0] * rows))
            for n in range(0, rows):
                self.chunks[j].set(n, 0, leading_block.chunks[j].get(n, 0))


class ConformalBlockVector:
    def __init__(self, dim, l, delta_12, delta_34, derivative_order, kept_pole_order, s_matrix, leading_block, pol_list, res_list):
        self.large_poles = []
        self.small_poles = []
        self.chunks = []

        nu = (dim / Integer(2)) - 1
        old_list = MeromorphicBlockVector(leading_block)
        for k in range(0, len(pol_list)):
            max_component = 0
            for j in range(0, len(leading_block.chunks)):
                for n in range(0, leading_block.chunks[j].nrows()):
                    max_component = max(max_component, abs(float(res_list[k].chunks[j].get(n, 0))))

            pole = delta_pole(nu, pol_list[k][1], l, pol_list[k][3])
            if max_component < cutoff:
                self.small_poles.append(pole)
            else:
                self.large_poles.append(pole)

        matrix = []
        if self.small_poles != []:
            for i in range(0, len(self.large_poles) // 2):
                for j in range(0, len(self.large_poles)):
                    matrix.append(1 / ((cutoff + unitarity_bound(dim, l) - self.large_poles[j]) ** (i + 1)))
            for i in range(0, len(self.large_poles) - (len(self.large_poles) // 2)):
                for j in range(0, len(self.large_poles)):
                    matrix.append(1 / (((1 / cutoff) - self.large_poles[j]) ** (i + 1)))
            matrix = DenseMatrix(len(self.large_poles), len(self.large_poles), matrix)

        for j in range(0, len(leading_block.chunks)):
            self.chunks.append(leading_block.chunks[j])

        for p in self.small_poles:
            vector = []
            for i in range(0, len(self.large_poles) // 2):
                vector.append(1 / ((unitarity_bound(dim, l) - p) ** (i + 1)))
            for i in range(0, len(self.large_poles) - (len(self.large_poles) // 2)):
                vector.append(1 / (((1 / cutoff) - p) ** (i + 1)))
            vector = DenseMatrix(len(self.large_poles), 1, vector)
            vector = matrix.solve(vector)

            k1 = self.get_pole_index(nu, l, pol_list, p)
            for i in range(0, len(self.large_poles)):
                k2 = self.get_pole_index(nu, l, pol_list, self.large_poles[i])
                for j in range(0, len(self.chunks)):
                    res_list[k2].chunks[j] = res_list[k2].chunks[j].add_matrix(res_list[k1].chunks[j].mul_scalar(vector.get(i, 0)))

        prod = 1
        for p in self.large_poles:
            k = self.get_pole_index(nu, l, pol_list, p)
            for j in range(0, len(self.chunks)):
                self.chunks[j] = self.chunks[j].mul_scalar(delta - p).add_matrix(res_list[k].chunks[j].mul_scalar(prod))
                for i in range(0, self.chunks[j].nrows()):
                    self.chunks[j].set(i, 0, self.chunks[j].get(i, 0).expand())
            prod *= delta - p
            prod = prod.expand()

        for j in range(0, len(self.chunks)):
            s_sub = s_matrix[0:derivative_order - j + 1, 0:derivative_order - j + 1]
            self.chunks[j] = s_sub.mul_matrix(self.chunks[j])

    def get_pole_index(self, nu, l, pol_list, p):
        for k in range(0, len(pol_list)):
            pole = delta_pole(nu, pol_list[k][1], l, pol_list[k][3])
            if abs(float(pole - p)) < tiny:
                return k
        return -1
