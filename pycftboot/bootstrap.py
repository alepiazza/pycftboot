#!/usr/bin/env python
"""
PyCFTBoot is an interface for the numerical bootstrap in arbitrary dimension,
a field that was initiated in 2008 by Rattazzi, Rychkov, Tonni and Vichi in
arXiv:0807.0004. Starting from the analytic structure of conformal blocks, the
code formulates semidefinite programs without any proprietary software. The
actual optimization step must be performed by David Simmons-Duffin's program
SDPB available at https://github.com/davidsd/sdpb.

PyCFTBoot may be used to find bounds on OPE coefficients and allowed regions in
the space of scaling dimensions for various CFT operators. All operators used in
the explicit correlators must be scalars, but they may have different scaling
dimensions and transform in arbitrary representations of a global symmetry.
"""
from typing import Union, List, Tuple
from functools import wraps
import xml.dom.minidom
import time
from symengine.lib.symengine_wrapper import (
    have_mpfr, RealMPFR, factorial, oo, uppergamma, log, DenseMatrix,
    pi, exp, sqrt
)

from .polynomial_vector import PolynomialVector
from .conformal_block_table import ConformalBlockTable
from .convolved_block_table import ConvolvedBlockTable
from .common import (
    rf, gather, deepcopy, unitarity_bound, coefficients, get_index,
)
from .constants import prec, delta, one, delta_ext, r_cross, zero, tiny
from .sdpb import sdpb_switch

if have_mpfr is False:
    print("Symengine must be compiled with MPFR support")
    quit(1)


# MPFR has no trouble calling gamma_inc quickly when the first argument is zero
# In case we need to go back to using non-zero values, the following might be faster
"""
import mpmath
mpmath.mp.dps = dec_prec
def uppergamma(x, a):
    return RealMPFR(str(mpmath.gammainc(mpmath.mpf(str(x)), a = mpmath.mpf(str(a)))), prec)
"""

SpinIrrep = Union[int, Tuple[int, Union[int, str]]]


class SDP:
    """
    A class where convolved conformal blocks are augmented by crossing equations
    which allow numerical bounds to be derived. All calls to `SDPB` happen through
    this class.

    Parameters
    ----------
    dim_list:        A list of all scaling dimensions that appear in the external
                     operators of the four-point functions being considered. If
                     there is only one, this may be a float instead of a list.
    conv_table_list: A list of all types of convolved conformal block tables that
                     appear in the crossing equations. If there is only one type,
                     this may be a `ConvolvedBlockTable` instance instead of a list.
    vector_types:    [Optional] A list of triples, one for each type of operator in
                     the sum rule. The third element of each triple is the arbitrary
                     label for that representation (something used to label
                     `PolynomialVector`s that are generated). The second element is
                     an even integer for even spin operators and an odd integer for
                     odd spin operators. The first element is everything else.
                     Specifically, it is a list of matrices of ordered quadruples
                     where a matrix is a list of lists. If the sum rule involves no
                     matrices, it may simply be a list of ordered quadruples. In a
                     quadruple, the first entry is a numerical coefficient and the
                     second entry is an index stating which element of
                     `conv_table_list` that coefficient should multiply. The third
                     and fourth entries (which may be omitted if `dim_list` has only
                     one entry) specify the external dimensions that should replace
                     `delta_ext` in a `ConvolvedConformalBlockTable` as positions in
                     `dim_list`. They are the "inner two" dimensions `j` and `k` if
                     convolved conformal blocks are given `i`, `j`, `k`, `l` labels
                     as in arXiv:1406.4858. The first triple must describe the even
                     spin singlet channel (where the identity can be found). After
                     this, the order of the triples is not important.
    prototype:       [Optional] A previous instance of `SDP` which may speed up the
                     allocation of this one. The idea is that if a bound on any
                     operator does not need to change from one table to the next,
                     the bilinear basis corresponding to it (which requires a
                     Cholesky decomposition and a matrix inversion to calculate)
                     might simply be copied.

    Attributes
    ----------
    dim:             The spatial dimension, inherited from `conv_block_table_list`.
    k_max:           The corresponding attribute from `conv_block_table_list`.
    l_max:           The corresponding attribute from `conv_block_table_list`.
    m_max:           The corresponding attribute from `conv_block_table_list`.
    n_max:           The corresponding attribute from `conv_block_table_list`.
    odd_spins:       Whether any element of `conv_block_table_list` has odd spins.
    table:           A list of matrices of `PolynomialVector`s where the number of
                     rows and columns is determined from `vector_types`. They are
                     ordered first by the type of representation and then by spin.
                     Each `PolynomialVector` may be longer than a `PolynomialVector`
                     from a single entry of `conv_block_table_list`. They represent
                     the concatenation of several such `PolynomialVectors`, one for
                     each row of a vectorial sum rule.
    m_order:         Analogous to `m_order` in `ConformalBlockTable` or
                     `ConvolvedBlockTable`, this keeps track of the number of `a`
                     derivatives in these longer `PolynomialVector`s.
    m_order:         Analogous to `n_order` in `ConformalBlockTable` or
                     `ConvolvedBlockTable`, this keeps track of the number of `b`
                     derivatives in these longer `PolynomialVector`s.
    options:         A list of strings where each string is a command line option
                     that will be passed when `SDPB` is run from this `SDP`. This
                     list should be touched with `set_option` and not directly.
    points:          In addition to `PolynomialVector`s whose entries allow `delta`
                     to take any positive value, the user may also include in the
                     sum rule `PolynomialVector`s whose entries are pure numbers.
                     In other words, she may evaluate some of them once and for all
                     at particular values of `delta` to force certain operators to
                     appear in the spectrum. This list should be touched with
                     `add_point` and not directly.
    unit:            A list which gives the `PolynomialVector` corresponding to the
                     identity. This is obtained simply by plugging `delta = 0` into
                     the zero spin singlet channel. If such a channel involves
                     matrices, the sum of all elements is taken since the conformal
                     blocks are normalized under the convention that all OPE
                     coefficients involving the identity are 1. It should not be
                     necessary to change this.
    irrep_set:       A list of ordered pairs, one for each type of operator in
                     `vector_types`. The second element of each is a label for the
                     representation. The first is a modified version of the first
                     matrix. The ordered quadruples do not correspond to the
                     prefactors and list positions anymore but to the four external
                     operator dimensions that couple to the block in this position.
                     It should not be necessary to change this.
    basis:           A list of matrices which has as many matrices as `table`.
                     Each triangular matrix stores a set of orthogonal polynomials
                     in the monomial basis. It should not be necessary to change
                     this.
    """

    def __init__(
            self, dim_list: Union[float, List[float]], conv_table_list: Union[ConformalBlockTable, List[ConformalBlockTable]],
            vector_types=[[[[[[1, 0, 0, 0]]]], 0, 0]], prototype=None,
            sdpb_mode: str = 'binary', sdpb_kwargs={}
    ):
        # If a user is looking at single correlators, we will not punish
        # her for only passing one dimension
        if not isinstance(dim_list, list):
            dim_list = [dim_list]
        if not isinstance(conv_table_list, list):
            conv_table_list = [conv_table_list]

        # Type checking
        if all([isinstance(tab, ConvolvedBlockTable) for tab in conv_table_list]):
            TypeError(f"conv_table_list = {conv_table_list} must be a ConvolvedBlockTable of a list of ConvolvedBlockTable")

        if prototype is not None:
            if not isinstance(prototype, SDP):
                TypeError(f"prototype = {prototype} must be a SDP object")

        # Same story here
        self.dim = 0
        self.k_max = 0
        self.l_max = 0
        self.m_max = 0
        self.n_max = 0
        self.odd_spins = False

        # Just in case these are different
        for tab in conv_table_list:
            self.dim = max(self.dim, tab.dim)
            self.k_max = max(self.k_max, tab.k_max)
            self.l_max = max(self.l_max, tab.l_max)
            self.m_max = max(self.m_max, tab.m_max)
            self.n_max = max(self.n_max, tab.n_max)

        self.points = []
        self.m_order = []
        self.n_order = []
        self.table = []
        self.unit = []
        self.irrep_set = []

        # Turn any "raw elements" from the vectorial sum rule into 1x1 matrices
        for i in range(0, len(vector_types)):
            for j in range(0, len(vector_types[i][0])):
                if not isinstance(vector_types[i][0][j][0], list):
                    vector_types[i][0][j] = [[vector_types[i][0][j]]]

        # Again, fill in arguments that need not be specified for single correlators
        for i in range(0, len(vector_types)):
            for j in range(0, len(vector_types[i][0])):
                for k in range(0, len(vector_types[i][0][j])):
                    for l in range(0, len(vector_types[i][0][j][k])):
                        if len(vector_types[i][0][j][k][l]) == 2:
                            vector_types[i][0][j][k][l].append(0)
                            vector_types[i][0][j][k][l].append(0)

        # We must assume the 0th element put in vector_types corresponds to the singlet channel
        # This is because we must harvest the identity from it
        for matrix in vector_types[0][0]:
            chosen_tab = conv_table_list[matrix[0][0][1]]

            for i in range(0, len(chosen_tab.table[0].vector)):
                unit = 0
                m = chosen_tab.m_order[i]
                n = chosen_tab.n_order[i]
                for r in range(0, len(matrix)):
                    for s in range(0, len(matrix[r])):
                        quad = matrix[r][s]
                        param = RealMPFR("0.5", prec) * (dim_list[quad[2]] + dim_list[quad[3]])
                        # tab = conv_table_list[quad[1]]
                        # factor = self.shifted_prefactor(tab.table[0].poles, r_cross, 0, 0)
                        # unit += factor * quad[0] * tab.table[0].vector[i].subs(delta, 0).subs(delta_ext, (dim_list[quad[2]] + dim_list[quad[3]]) / 2.0)
                        unit += 2 * quad[0] * (RealMPFR("0.25", prec) ** param) * rf(-param, n) * rf(2 * n - 2 * param, m) / (factorial(m) * factorial(n))

                self.m_order.append(m)
                self.n_order.append(n)
                self.unit.append(unit)

        # Looping over types and spins gives "0 - S", "0 - T", "1 - A" and so on
        for vec in vector_types:
            # Instead of specifying even or odd spins, the user can specify a list of spins
            if isinstance(vec[1], list):
                spin_list = vec[1]
            elif (vec[1] % 2) == 1:
                self.odd_spins = True
                spin_list = range(1, self.l_max, 2)
            else:
                spin_list = range(0, self.l_max, 2)

            for l in spin_list:
                size = len(vec[0][0])

                outer_list = []
                for r in range(0, size):
                    inner_list = []
                    for s in range(0, size):
                        derivatives = []
                        large_poles = []
                        for matrix in vec[0]:
                            quad = matrix[r][s]
                            tab = conv_table_list[quad[1]]

                            if tab.odd_spins:
                                index = l
                            else:
                                index = l // 2
                            if quad[0] != 0:
                                large_poles = tab.table[index].poles

                            for i in range(0, len(tab.table[index].vector)):
                                derivatives.append(quad[0] * tab.table[index].vector[i].subs(delta_ext, (dim_list[quad[2]] + dim_list[quad[3]]) / 2.0))
                        inner_list.append(PolynomialVector(derivatives, [l, vec[2]], large_poles))
                    outer_list.append(inner_list)
                self.table.append(outer_list)

        # We are done with vector_types now so we can change it
        for vec in vector_types:
            matrix = deepcopy(vec[0][0])
            for r in range(0, len(matrix)):
                for s in range(0, len(matrix)):
                    quad = matrix[r][s]
                    dim2 = dim_list[quad[2]]
                    dim3 = dim_list[quad[3]]
                    dim1 = dim2 + conv_table_list[quad[1]].delta_12
                    dim4 = dim3 - conv_table_list[quad[1]].delta_34
                    matrix[r][s] = [dim1, dim2, dim3, dim4]
            self.irrep_set.append([matrix, vec[2]])

        self.bounds = [0.0] * len(self.table)
        self.options = []

        if prototype is None:
            self.basis = [0] * len(self.table)
            self.set_bound(reset_basis=True)
        else:
            self.basis = []
            for mat in prototype.basis:
                self.basis.append(mat)
            self.set_bound(reset_basis=False)

        self.sdpb = sdpb_switch(sdpb_mode, sdpb_kwargs)
        self.sdpb_mode = sdpb_mode
        self.sdpb_kwargs = sdpb_kwargs

    def add_point(self, spin_irrep: SpinIrrep = -1, dimension: float = -1, extra: list = []):
        """
        Tells the `SDP` that a particular fixed operator should be included in the
        sum rule. If called with one argument, all points with that label will be
        removed. If called with no arguments, all points with any label will be
        removed.

        Parameters
        ----------
        spin_irrep: [Optional] An ordered pair used to label the `PolynomialVector`
                    for the operator. The first entry is the spin, the second is the
                    label which must be found in `vector_types` or 0 if not present.
                    Defaults to -1 which means all operators.
        dimension:  [Optional] The scaling dimension of the operator being added.
                    Defaults to -1 which means the point should be removed.
        extra:      [Optional] A list of quintuples specifying information about
                    other operators that should be packaged with this operator. The
                    first two elements of a quintuple are the `spin_irrep` and
                    `dimension` except for the operator which is not being added
                    separately because its presence is tied to this one. The next
                    two elements of a quintuple are ordered pairs giving positions
                    in the crossing equation matrices. The operator described by the
                    first two quintuple elements should have its contribution in the
                    position given by the first ordered pair added to that of the
                    operator described by `spin_irrep` and `dimension` in the
                    position given by the second ordered pair. The final element of
                    the quintuple is a coefficient that should multiply whatever is
                    added. The purpose of this is to enforce OPE coefficient
                    relations as in arXiv:1603.04436.
        """
        if spin_irrep == -1:
            self.points = []
        else:
            if isinstance(spin_irrep, int):
                spin_irrep = [spin_irrep, 0]
            if dimension != -1:
                self.points.append((spin_irrep, dimension, extra))
            else:
                for p in self.points:
                    if p[0] == spin_irrep:
                        self.points.remove(p)

    def get_bound(self, gapped_spin_irrep: SpinIrrep):
        """
        Returns the minimum scaling dimension of a given operator in this `SDP`.
        This will return the unitarity bound until the user starts calling
        `set_bound`.

        Parameters
        ----------
        gapped_spin_irrep: An ordered pair used to label the `PolynomialVector`
                           whose bound should be read. The first entry is the spin
                           and the second is the label found in `vector_types` or
                           0 if not present.
        """
        if isinstance(gapped_spin_irrep, int):
            gapped_spin_irrep = [gapped_spin_irrep, 0]
        for l in range(0, len(self.table)):
            if self.table[l][0][0].label == gapped_spin_irrep:
                return self.bounds[l]

    def set_bound(self, gapped_spin_irrep: SpinIrrep = -1, delta_min: float = -1, reset_basis=True):
        """
        Sets the minimum scaling dimension of a given operator in the sum rule. If
        called with one argument, the operator with that label will be assigned the
        unitarity bound. If called with no arguments, all operators will be assigned
        the unitarity bound.

        Parameters
        ----------
        gapped_spin_irrep: [Optional] An ordered pair used to label the
                           `PolynomialVector` whose bound should be set. The first
                           entry is the spin and the second is the label found in
                           `vector_types` or 0 if not present. Defaults to -1 which
                           means all operators.
        delta_min:         [Optional] The minimum scaling dimension to set. Also
                           accepts oo to indicate that a continuum should not be
                           included. Defaults to -1 which means unitarity.
        reset_basis:       [Optional] An internal parameter which may be used to
                           prevent the orthogonal polynomials which improve the
                           numerical stability of `SDPB` from being recalculated.
                           Defaults to `True`.
        """
        if gapped_spin_irrep == -1:
            for l in range(0, len(self.table)):
                spin = self.table[l][0][0].label[0]
                self.bounds[l] = unitarity_bound(self.dim, spin)

                if reset_basis:
                    self.set_basis(l)
        else:
            if isinstance(gapped_spin_irrep, int):
                gapped_spin_irrep = [gapped_spin_irrep, 0]

            l = self.get_table_index(gapped_spin_irrep)
            spin = gapped_spin_irrep[0]

            if delta_min == -1:
                self.bounds[l] = unitarity_bound(self.dim, spin)
            else:
                self.bounds[l] = delta_min

            if reset_basis and delta_min != oo:
                self.set_basis(l)

    def get_table_index(self, spin_irrep: SpinIrrep):
        """
        Searches for the label of a `PolynomialVector` and returns its position in
        `table` or -1 if not found.

        Parameters
        ----------
        spin_irrep: An ordered pair of the type passed to `set_bound`. Used to
                    label the spin and representation being searched.
        """
        if isinstance(spin_irrep, int):
            spin_irrep = [spin_irrep, 0]
        for l in range(0, len(self.table)):
            if self.table[l][0][0].label == spin_irrep:
                return l
        return -1

    def set_basis(self, index: int):
        """
        Calculates a basis of polynomials that are orthogonal with respect to the
        positive measure prefactor that turns a `PolynomialVector` into a rational
        approximation to a conformal block. It should not be necessary to explicitly
        call this.

        Parameters
        ----------
        index: The position of the matrix in `table` whose basis needs updating.
        """
        poles = self.table[index][0][0].poles
        delta_min = self.bounds[index]
        delta_min = float(delta_min)
        delta_min = RealMPFR(str(delta_min), prec)
        bands = []
        matrix = []

        degree = 0
        size = len(self.table[index])
        for r in range(0, size):
            for s in range(0, size):
                polynomial_vector = self.table[index][r][s].vector

                for n in range(0, len(polynomial_vector)):
                    expression = polynomial_vector[n].expand()
                    degree = max(degree, len(coefficients(expression)) - 1)

        # Separate the poles and associate each with an uppergamma function
        # This avoids computing these same functions for each d in the loop below
        gathered_poles = gather(poles)
        poles = []
        orders = []
        gammas = []
        for p in gathered_poles:
            if p < delta_min:
                poles.append(p - delta_min)
                orders.append(gathered_poles[p])
                gammas.append(uppergamma(zero, (p - delta_min) * log(r_cross)))

        for d in range(0, 2 * (degree // 2) + 1):
            result = (r_cross ** delta_min) * self.integral(d, poles, orders, gammas)
            bands.append(result)
        for r in range(0, (degree // 2) + 1):
            new_entries = []
            for s in range(0, (degree // 2) + 1):
                new_entries.append(bands[r + s])
            matrix.append(new_entries)

        matrix = DenseMatrix(matrix)
        matrix = matrix.cholesky()
        matrix = matrix.inv()
        self.basis[index] = matrix

    def reshuffle_with_normalization(self, vector, norm):
        """
        Converts between the Mathematica definition and the bootstrap definition of
        an SDP. As explained in arXiv:1502.02033, it is natural to normalize the
        functionals being found by demanding that they give 1 when acting on a
        particular `PolynomialVector`. `SDPB` on the other hand works with
        functionals that have a fixed leading component. This is an equivalent
        problem after a trivial reshuffling.

        Parameters
        ----------
        vector: The `vector` part of the `PolynomialVector` needing to be shuffled.
        norm:   The `vector` part of the `PolynomialVector` constrained to have
                unit action under the functional before the reshuffling.
        """
        norm_hack = []
        for el in norm:
            norm_hack.append(float(el))

        max_index = norm_hack.index(max(norm_hack, key=abs))
        const = vector[max_index] / norm[max_index]
        ret = []

        for i in range(0, len(norm)):
            ret.append(vector[i] - const * norm[i])

        ret = [const] + ret[:max_index] + ret[max_index + 1:]
        return ret

    def short_string(self, num):
        """
        Returns the string representation of a number except with an attempt to trim
        superfluous zeros if the number is too small.

        Parameters
        ----------
        num: The number.
        """
        if abs(num) < tiny:
            return "0"
        else:
            return str(num)

    def make_laguerre_points(self, degree):
        """
        Returns a list of convenient sample points for the XML files of `SDPB`.

        Parameters
        ----------
        degree: The maximum degree of all polynomials in a `PolynomialVector`.
        """
        ret = []
        for d in range(0, degree + 1):
            point = -(pi.n(prec) ** 2) * ((4 * d - 1) ** 2) / (64 * (degree + 1) * log(r_cross))
            ret.append(point)
        return ret

    def shifted_prefactor(self, poles, base, x, shift):
        """
        Returns the positive measure prefactor that turns a `PolynomialVector` into
        a rational approximation to a conformal block. Evaluating this at a sample
        point produces a sample scaling needed by the XML files of `SDPB`.

        Parameters
        ----------
        poles: The roots of the prefactor's denominator, often from the `poles`
               attribute of a `PolynomialVector`.
        base:  The base of the exponential in the numerator, often the crossing
               symmetric value of the radial co-ordinate.
        x:     The argument of the function, often `delta`.
        shift: An amount by which to shift `x`. This should match one of the minimal
               values assigned by `set_bound`.
        """
        product = 1
        for p in poles:
            product *= x - (p - shift)
        return (base ** (x + shift)) / product

    def basic_integral(self, pos, pole, order, gamma_val):
        """
        Returns the inner product of two monic monomials with respect to a more
        basic positive measure prefactor which has just a single pole.

        Parameters
        ----------
        pos:       The sum of the degrees of the two monomials.
        pole:      The root of the prefactor's denominator.
        order:     The multiplicity of this root.
        gamma_val: The associated incomplete gamma function. Note that it is no
                   longer uppergamma(0, pole * log(r_cross)) because we are
                   performing a change of variables.
        """
        if order == 1:
            ret = exp(-pole) * (pole ** pos) * gamma_val
            for i in range(0, pos):
                ret += factorial(pos - i - 1) * (pole ** i)
            return ret
        elif pos == 0:
            return ((-pole) ** (1 - order) / (order - 1)) - (one / (order - 1)) * self.basic_integral(pos, pole, order - 1, gamma_val)
        else:
            return (one / (order - 1)) * (pos * self.basic_integral(pos - 1, pole, order - 1, gamma_val) - self.basic_integral(pos, pole, order - 1, gamma_val))

    def integral(self, pos, poles, orders, gammas):
        """
        Returns the inner product of two monic monomials with respect to the
        positive measure prefactor that turns a `PolynomialVector` into a rational
        approximation to a conformal block.

        Parameters
        ----------
        pos:    The sum of the degrees of the two monomials.
        poles:  A list of the roots of the prefactor's denominator.
        orders: The multiplicities of those poles.
        gammas: A list representing the image of `poles` under the map which sends
                x to uppergamma(0, x * log(r_cross)).
        """
        ret = zero
        if len(poles) == 0:
            return factorial(pos) / ((-log(r_cross)) ** (pos + 1))

        for i in range(0, len(poles)):
            pole = poles[i]
            order = orders[i]
            gamma_val = gammas[i]
            other_poles = poles[:i] + poles[i + 1:]
            other_orders = orders[:i] + orders[i + 1:]
            exponents = {}
            exponents[tuple(other_orders)] = 1
            # For an order 3 pole, 0, 1 and 2 derivatives are needed in the Laurent series
            for j in range(0, order):
                for term in exponents:
                    prod = factorial(j)
                    for k in range(0, len(term)):
                        prod *= (pole - other_poles[k]) ** term[k]
                    ret += (one / prod) * exponents[term] * ((-log(r_cross)) ** (order - j - 1 - pos)) * self.basic_integral(pos, -pole * log(r_cross), order - j, gamma_val)
                if j == order - 1:
                    break
                # Update exponents to move onto the next derivative
                new_exponents = {}
                for term in exponents:
                    for k in range(0, len(term)):
                        new_term = list(term)
                        new_term[k] += 1
                        new_term = tuple(new_term)
                        if new_term in new_exponents:
                            new_exponents[new_term] += -term[k] * exponents[term]
                        else:
                            new_exponents[new_term] = -term[k] * exponents[term]
                exponents = new_exponents

        return ret

    def write_xml(self, obj, norm, name="mySDP"):
        """
        Outputs an XML file describing the `table`, `bounds`, `points` and `basis`
        for this `SDP` in a format that `SDPB` can use to check for solvability.
        If the user has the Elemental version of `SDPB` then the `pvm2sdb` utility
        (assumed to be in the same directory) is also run.

        Parameters
        ----------
        obj:  Objective vector (often the `vector` part of a `PolynomialVector`)
              whose action under the found functional should be maximized.
        norm: Normalization vector (often the `vector` part of a `PolynomialVector`)
              which should have unit action under the functionals.
        name: [Optional] Name of the XML file to produce without any ".xml" at the
              end. Defaults to "mySDP".
        """
        obj = self.reshuffle_with_normalization(obj, norm)
        laguerre_points = []
        laguerre_degrees = []
        extra_vectors = []
        degree_sum = 0

        # Handle discretely added points
        for p in self.points:
            l = self.get_table_index(p[0])
            size = len(self.table[l])

            outer_list = []
            for r in range(0, size):
                inner_list = []
                for s in range(0, size):
                    new_vector = []
                    for i in range(0, len(self.table[l][r][s].vector)):
                        addition = self.table[l][r][s].vector[i].subs(delta, p[1])
                        for quint in p[2]:
                            if quint[3][0] != r or quint[3][1] != s:
                                continue
                            l_new = self.get_table_index(quint[0])
                            r_new = quint[2][0]
                            s_new = quint[2][1]
                            coeff = quint[4]
                            coeff *= self.shifted_prefactor(self.table[l_new][0][0].poles, r_cross, quint[1], 0)
                            coeff /= self.shifted_prefactor(self.table[l][0][0].poles, r_cross, p[1], 0)
                            addition += coeff * self.table[l_new][r_new][s_new].vector[i].subs(delta, quint[1])
                        new_vector.append(addition)
                    inner_list.append(PolynomialVector(new_vector, p[0], self.table[l][r][s].poles))
                outer_list.append(inner_list)
            extra_vectors.append(outer_list)
        self.table += extra_vectors

        doc = xml.dom.minidom.Document()
        root_node = doc.createElement("sdp")
        doc.appendChild(root_node)

        objective_node = doc.createElement("objective")
        matrices_node = doc.createElement("polynomialVectorMatrices")
        root_node.appendChild(objective_node)
        root_node.appendChild(matrices_node)

        # Here, we use indices that match the SDPB specification
        for n in range(0, len(obj)):
            elt_node = doc.createElement("elt")
            elt_node.appendChild(doc.createTextNode(self.short_string(obj[n])))
            objective_node.appendChild(elt_node)

        for j in range(0, len(self.table)):
            if j >= len(self.bounds):
                delta_min = 0
            else:
                delta_min = self.bounds[j]
            if delta_min == oo:
                continue
            size = len(self.table[j])
            degree = 0

            matrix_node = doc.createElement("polynomialVectorMatrix")
            rows_node = doc.createElement("rows")
            cols_node = doc.createElement("cols")
            elements_node = doc.createElement("elements")
            sample_point_node = doc.createElement("samplePoints")
            sample_scaling_node = doc.createElement("sampleScalings")
            bilinear_basis_node = doc.createElement("bilinearBasis")
            rows_node.appendChild(doc.createTextNode(size.__str__()))
            cols_node.appendChild(doc.createTextNode(size.__str__()))

            for r in range(0, size):
                for s in range(0, size):
                    polynomial_vector = self.reshuffle_with_normalization(self.table[j][r][s].vector, norm)
                    vector_node = doc.createElement("polynomialVector")

                    for n in range(0, len(polynomial_vector)):
                        expression = polynomial_vector[n].expand()
                        # Impose unitarity bounds and the specified gap
                        expression = expression.subs(delta, delta + delta_min).expand()
                        coeff_list = coefficients(expression)
                        degree = max(degree, len(coeff_list) - 1)

                        polynomial_node = doc.createElement("polynomial")
                        for coeff in coeff_list:
                            coeff_node = doc.createElement("coeff")
                            coeff_node.appendChild(doc.createTextNode(self.short_string(coeff)))
                            polynomial_node.appendChild(coeff_node)
                        vector_node.appendChild(polynomial_node)
                    elements_node.appendChild(vector_node)

            poles = self.table[j][0][0].poles
            index = get_index(laguerre_degrees, degree)

            if j >= len(self.bounds):
                points = [self.points[j - len(self.bounds)][1]]
            elif index == -1:
                points = self.make_laguerre_points(degree)
                laguerre_points.append(points)
                laguerre_degrees.append(degree)
            else:
                points = laguerre_points[index]

            for d in range(0, degree + 1):
                elt_node = doc.createElement("elt")
                elt_node.appendChild(doc.createTextNode(points[d].__str__()))
                sample_point_node.appendChild(elt_node)
                damped_rational = self.shifted_prefactor(poles, r_cross, points[d], RealMPFR(str(delta_min), prec))
                elt_node = doc.createElement("elt")
                elt_node.appendChild(doc.createTextNode(damped_rational.__str__()))
                sample_scaling_node.appendChild(elt_node)

            matrix = []
            if j >= len(self.bounds):
                result = self.shifted_prefactor(poles, r_cross, points[0], zero)
                result = one / sqrt(result)
                matrix = DenseMatrix([[result]])
            else:
                matrix = self.basis[j]

            for d in range(0, (degree // 2) + 1):
                polynomial_node = doc.createElement("polynomial")
                for q in range(0, d + 1):
                    coeff_node = doc.createElement("coeff")
                    coeff_node.appendChild(doc.createTextNode(matrix[d, q].__str__()))
                    polynomial_node.appendChild(coeff_node)
                bilinear_basis_node.appendChild(polynomial_node)

            matrix_node.appendChild(rows_node)
            matrix_node.appendChild(cols_node)
            matrix_node.appendChild(elements_node)
            matrix_node.appendChild(sample_point_node)
            matrix_node.appendChild(sample_scaling_node)
            matrix_node.appendChild(bilinear_basis_node)
            matrices_node.appendChild(matrix_node)
            degree_sum += degree + 1

        # Recognize an SDP that looks overdetermined
        if degree_sum < len(self.unit):
            raise RuntimeWarning("Crossing equations have too many derivative components")

        self.table = self.table[:len(self.bounds)]

        with open(f"{name}.xml", "w") as xml_file:
            doc.writexml(xml_file, addindent="    ", newl='\n')

        doc.unlink()

        if self.sdpb.version == 2:
            self.sdpb.pvm2sdp_run(f"{name}.xml", name)

    def manage_name(method):
        @wraps(method)
        def wrapper(self, *args, **kwargs):
            name_is_set = 'name' in kwargs
            sdpDir_already_exist = "sdpDir" in self.sdpb.options

            if name_is_set:
                if sdpDir_already_exist:
                    old_name = self.sdpb.get_option["sdpDir"]
                self.sdpb.set_option("sdpDir", kwargs['name'])

            output = method(self, *args, **kwargs)

            if name_is_set:
                if sdpDir_already_exist:
                    self.sdpb.set_option("sdpDir", old_name)
                else:
                    self.sdpb.options.pop("sdpDir")

            return output

        return wrapper

    @manage_name
    def iterate(self, *, name: str = None):
        """
        Returns `True` if this `SDP` with its current gaps represents an allowed CFT
        and `False` otherwise.

        Parameters
        ----------
        name:       [Optional] The name of the XML file generated in the process
                    without any ".xml" at the end. Defaults to "mySDP".
        """
        # Prepare SDPB input
        obj = [0.0] * len(self.table[0][0][0].vector)
        self.write_xml(obj, self.unit, self.sdpb.get_option("sdpDir"))

        # Run SDPB
        extra_options = {'findPrimalFeasible': True, 'findDualFeasible': True}
        if self.sdpb.version == 1:
            extra_options["noFinalCheckpoint"] = True

        self.sdpb.run(extra_options)

        output = self.sdpb.read_output(self.sdpb.get_option("outDir"))

        terminate_reason = output["terminateReason"]
        allowed = terminate_reason == "found primal feasible solution"

        print(f'{"allowed" if allowed else "not allowed"} ({terminate_reason})')

        return allowed

    @manage_name
    def bisect(self, lower, upper, threshold, spin_irrep, isolated=False, reverse=False, bias=None, *, name: str = None):
        """
        Uses a binary search to find the maximum allowed gap in a particular type
        of operator before the CFT stops existing. The allowed value closest to the
        boundary is returned.

        Parameters
        ----------
        lower:      A scaling dimension for the operator known to be allowed.
        upper:      A scaling dimension for the operator known to be disallowed.
        threshold:  How accurate the bisection needs to be before returning.
        spin_irrep: An ordered pair of the type passed to `set_bound`. Used to
                    label the spin and representation of the operator whose
                    dimension is being bounded.
        isolated:   [Optional] Whether to bisect the position of an isolated
                    operator rather than the gap where the continuum starts.
                    Defaults to `False`.
        reverse:    [Optional] Whether we are looking for a lower bound instead of
                    an upper bound. This should only be used when `isolated` is
                    `True`. Defaults to `False`.
        bias:       [Optional] The ratio between the expected time needed to rule
                    out a CFT and the expected time needed to conclude that it
                    cannot be. Defaults to `None` which means that this will be
                    measured as the binary search progresses.
        """
        self.sdpb.remove_log()

        x = 0.5
        d_time = 0
        p_time = 0
        bias_found = False
        checkpoints = False
        old = self.get_bound(spin_irrep)
        if bias is not None:
            bias = min(bias, 1.0 / bias)

        while abs(upper - lower) > threshold:
            if bias is None and d_time != 0 and p_time != 0:
                bias = p_time / d_time
            if bias is not None and bias_found is False:
                # Bisection within a bisection
                u = 0.5
                l = 0.0
                while abs(u - l) > 0.01:
                    x = (u + l) / 2.0
                    frac = log((x ** x) * ((1 - x) ** (1 - x))) / log(x / (1 - x))
                    test = (frac - x) / (frac - x + 1)
                    if test > bias:
                        u = x
                    else:
                        l = x
                bias_found = True

            test = lower + x * (upper - lower)
            print(f'Trying {test}: ', end='', flush=True)
            if isolated is True:
                self.add_point(spin_irrep, test)
            else:
                self.set_bound(spin_irrep, test)

            # Using the same name twice in a row is only dangerous if the runs are really long
            start = time.time()
            if checkpoints and self.sdpb.version == 1:
                result = self.iterate(name=str(start))
            else:
                result = self.iterate()
            end = time.time()
            if int(end - start) > int(self.sdpb.get_option("checkpointInterval")):
                checkpoints = True
            if isolated is True:
                self.points = self.points[:-1]

            if result is False:
                if reverse is False:
                    upper = test
                else:
                    lower = test
                d_time = end - start
            else:
                if reverse is False:
                    lower = test
                else:
                    upper = test
                p_time = end - start

        self.set_bound(spin_irrep, old)
        if reverse is False:
            return lower
        else:
            return upper

    @manage_name
    def opemax(self, dimension, spin_irrep, reverse=False, vector=None, *, name: str = None):
        """
        Minimizes or maximizes the squared length of the vector of OPE coefficients
        involving an operator with a prescribed scaling dimension, spin and global
        symmetry representation. This results in a matrix produced by the action of
        the functional found by `SDPB`. If a direction in OPE space has been passed
        then the corresponding matrix element is returned. Otherwise, the matrix is
        returned and it is up to the user to find the unconstrained minimum or
        maximum value by diagonalizing it.

        Parameters
        ----------
        dimension:  The scaling dimension of the operator whose OPE coefficients
                    are having their length being bounded.
        spin_irrep: An ordered pair of the type passed to `set_bound`. Used to label
                    the spin and representation of the operator whose OPE
                    coefficients have their length being bounded.
        reverse:    [Optional] Whether to minimize a squared OPE coefficient vector
                    instead of maximizing it. This only has a chance of working if
                    the bounds are such that the specified operator is isolated.
                    Defaults to `False`.
        vector:     [Optional] A unit vector specifying the direction in OPE space
                    being scanned if applicable. In a 2x2 scan, for instance, which
                    is specified by one angle, the components of this vector will
                    be the sine and the cosine. Defaults to `None`.
        name:       [Optional] Name of the XML file generated in the process without
                    any ".xml" at the end. Defaults to "mySDP".
        """
        l = self.get_table_index(spin_irrep)
        size = len(self.table[l])
        if reverse:
            sign = -1
        else:
            sign = 1
        prod = self.shifted_prefactor(self.table[l][0][0].poles, r_cross, dimension, 0) * sign

        if vector is None or len(vector) != size:
            vec = [0] * size
            vec[0] = 1
        else:
            vector_length = 0
            for r in range(0, size):
                vector_length += vector[r] ** 2
            vector_length = sqrt(vector_length)
            for s in range(0, size):
                vec[s] = vector[s] / vector_length

        norm = []
        for i in range(0, len(self.unit)):
            el = 0
            for r in range(0, size):
                for s in range(0, size):
                    el += vec[r] * vec[s] * self.table[l][r][s].vector[i].subs(delta, dimension)
            norm.append(el * prod)
        functional = self.solution_functional(self.get_bound(spin_irrep), spin_irrep, self.unit, norm)
        output = self.sdpb.read_output(self.sdpb.get_option("outDir"))
        primal_value = output["primalObjective"]
        if size == 1 or vector is not None:
            return float(primal_value) * (-1)

        # This primal value will be divided by 1 or something different if the matrix is not 1x1
        outer_list = []
        for r in range(0, size):
            inner_list = []
            for s in range(0, size):
                inner_product = 0.0
                polynomial_vector = self.reshuffle_with_normalization(self.table[l][r][s].vector, norm)

                for i in range(0, len(self.table[l][r][s].vector)):
                    inner_product += functional[i] * polynomial_vector[i]
                    inner_product = inner_product.subs(delta, dimension)

                inner_list.append(float(inner_product))
            outer_list.append(inner_list)
        if reverse:
            print("Divide " + str(float(primal_value)) + " by the maximum eigenvalue")
        else:
            print("Divide " + str(float(primal_value)) + " by the minimum eigenvalue")
        return DenseMatrix(outer_list)

    @manage_name
    def solution_functional(self, dimension, spin_irrep, obj=None, norm=None, *, name: str = None):
        """
        Returns a functional (list of numerical components) that serves as a
        solution to the `SDP`. Like `iterate`, this sets a bound, generates an XML
        file and calls `SDPB`. However, rather than stopping after it determines
        that the `SDP` is indeed solvable, it will finish the computation to find
        the actual functional.

        Parameters
        ----------
        dimension:  The minimum value of the scaling dimension to test.
        spin_irrep: An ordered pair of the type passed to `set_bound`. Used to label
                    the spin / representation of the operator being given a minimum
                    scaling dimension of `dimension`.
        obj:        [Optional] The objective vector whose action under the found
                    functional should be maximized. Defaults to `None` which means
                    it will be determined automatically just like it is in
                    `iterate`.
        norm:       [Optional] Normalization vector which should have unit action
                    under the functional. Defaults to `None` which means it will be
                    determined automatically just like it is in `iterate`.
        name:       [Optional] The name of the XML file generated in the process
                    without any ".xml" at the end. Defaults to "mySDP".
        """
        if obj is None:
            obj = [0.0] * len(self.table[0][0][0].vector)
        if norm is None:
            norm = self.unit

        old = self.get_bound(spin_irrep)
        self.set_bound(spin_irrep, dimension)
        self.write_xml(obj, norm, self.sdpb.get_option("sdpDir"))
        self.set_bound(spin_irrep, old)

        self.sdpb.run({"noFinalCheckpoint": True})

        output = self.sdpb.read_output(self.sdpb.get_option("outDir"))

        return [one] + output["y"]

    # THIS FUNCTION IS DEPRECATED SINCE I DON'T HAVE THE OUTPUT TO REWRITE IT IN A BETTER WAY
    # def convert_spectrum_file(self, input_path, output_path, rescaling=4 ** delta):
    #     """
    #     Reads a spectrum produced by the arXiv:1603.04444 script and outputs a file
    #     with physical dimensions and OPE coefficients. Instead of a scaling
    #     dimension, the original file reports the difference between the scaling
    #     dimension and the gap. Instead of an OPE coefficient, the original file
    #     reports the factor relating the OPE coefficient to the positive prefactor.
    #     Note that this only works if `set_bound` has not been called since the
    #     XML file was generated.

    #     Parameters
    #     ----------
    #     input_path:  The path to the spectrum in Mathematica-like format.
    #     output_path: The path desired for the file after the additive and
    #                  multiplicative corrections have been performed.
    #     rescaling:   [Optional] An expression, which may depend on `delta` and
    #                  `ell`, for changing the convention used for OPE coefficients.
    #                  Defaults to 4 ** delta.
    #     """
    #     in_file = open(input_path, 'r')
    #     out_file = open(output_path, 'w')

    #     out_file.write('{')
    #     for j in range(0, len(self.table) + len(self.points)):
    #         if j >= len(self.table):
    #             shift = self.points[j - len(self.table)][1]
    #             spin = self.points[j - len(self.table)][0][0]
    #         else:
    #             shift = self.bounds[j]
    #             spin = self.table[j][0][0].label[0]
    #         out_file.write(str(j) + " -> ")
    #         line = next(in_file)[:-2].split("->")[1]
    #         line = line.replace('{', '[').replace('}', ']')
    #         line = re.sub("([0-9]+\.[0-9]+e?-?[0-9]+)", r"RealMPFR('\1', prec)", line)
    #         exec("ops = " + line)
    #         for o in range(0, len(ops)):
    #             ops[o][0] = ops[o][0] + shift
    #             if j >= len(self.table):
    #                 prod = 1
    #             else:
    #                 prod = self.shifted_prefactor(self.table[j][0][0].poles, r_cross, ops[o][0], 0)
    #             if "subs" in rescaling:
    #                 prod *= rescaling.subs(delta, ops[o][0]).subs(ell, spin)
    #             else:
    #                 prod *= rescaling
    #             for t in range(0, len(ops[o][1])):
    #                 ops[o][1][t] = ops[o][1][t] / sqrt(prod)
    #         ops_str = str(ops).replace('[', '{').replace(']', '}')
    #         out_file.write(ops_str + ",\n")
    #     # Copy the objective at the end
    #     out_file.write(next(in_file))

    #     in_file.close()
    #     out_file.close()

    def extremal_dimensions(self, functional, spin_irrep, zero_threshold, tmp_file):
        """
        When a functional acts on `PolynomialVector`s, this finds approximate zeros
        of the resulting expression with the `unisolve` executable. When the sum
        rule has matrices of `PolynomialVector`s, sufficiently small local minima
        of their determinants are returned. The list consists of dimensions for a
        given spin and representation. The logic is a subset of that used in the
        arXiv:1603.04444 script.

        Parameters
        ----------
        functional:     A list of functional components of the type returned by
                        `solution_functional`.
        spin_irrep:     An ordered pair used to label the type of operator whose
                        extremal dimensions are being found. The first entry is the
                        spin and the second entry is the representation label found
                        in `vector_types`.
        zero_threshold: The threshold for identifying a real zero. The determinant
                        over its second derivative must be less than this value.
        tmp_file:       A temporary file name where to put the `unisolve` input
        """
        zeros = []
        entries = []
        l = self.get_table_index(spin_irrep)

        size = len(self.table[l])
        for r in range(0, size):
            for s in range(0, size):
                inner_product = 0.0
                polynomial_vector = self.reshuffle_with_normalization(self.table[l][r][s].vector, self.unit)

                for i in range(0, len(self.table[l][r][s].vector)):
                    inner_product += functional[i] * polynomial_vector[i]
                    inner_product = inner_product.expand()

                entries.append(inner_product)

        matrix = DenseMatrix(size, size, entries)
        det0 = matrix.det().expand()
        det1 = det0.diff(delta)
        det2 = det1.diff(delta)
        coeffs = coefficients(det1)
        # Pass output to unisolve
        with open(tmp_file, "w") as pol_file:
            pol_file.write("drf\n")
            pol_file.write(str(prec) + "\n")
            pol_file.write(str(len(coeffs) - 1) + "\n")
            for c in coeffs:
                pol_file.write(str(c) + "\n")

        unisolve_proc = self.sdpb.unisovle_run(prec, tmp_file)
        spec_lines = unisolve_proc.stdout.splitlines()
        for line in spec_lines:
            pair = line.replace('(', '').replace(')', '').split(',')
            real = RealMPFR(pair[0], prec)
            imag = RealMPFR(pair[1], prec)
            if imag < tiny and det0.subs(delta, real) / det2.subs(delta, real) < zero_threshold:
                zeros.append(real)
        return zeros

    def extremal_coefficients(self, dimensions, spin_irreps, tmp_name, nullity=1):
        """
        Once the full extremal spectrum is known, one can reconstruct the OPE
        coefficients that cause those convolved conformal blocks to sum to the
        `SDP`'s `unit`. This outputs a vector of squared OPE coefficients
        determined in this way.

        Parameters
        ----------
        dimensions:  A list of dimensions in the spectrum as returned by
                     `extremal_dimensions`. However, it must be the union of such
                     scaling dimensions over all possible `spin_irrep` inputs to
                     `extremal_dimensions`.
        spin_irreps: A list of ordered pairs of the type passed to
                     `extremal_dimensions` used to label the spin and global
                     symmetry representations of all operators that
                     `extremal_dimensions` can find. This list must be in the same
                     order used for `dimensions`.
        nullity:     [Optional] The number of extra equations to use beyond the
                     number of unknown variables. If this is non-zero, a positivity
                     constraint will be placed on the optimal OPE coefficients.
                     Defaults to 1.
        tmp_name:    A temporary sdpDir used internally by `least_absolute_distance`
        """
        # Builds an auxillary table to store the specific vectors in this sum rule
        extremal_table = []
        zeros = min(len(dimensions), len(spin_irreps))
        for j in range(0, zeros):
            if isinstance(spin_irreps[j], int):
                spin_irreps[j] = [spin_irreps[j], 0]
            l = self.get_table_index(spin_irreps[j])
            factor = self.shifted_prefactor(self.table[l][0][0].poles, r_cross, dimensions[j], 0)
            size = len(self.table[l])
            outer_list = []
            for r in range(0, size):
                inner_list = []
                for s in range(0, size):
                    extremal_entry = []
                    for i in range(0, len(self.unit)):
                        extremal_entry.append(self.table[l][r][s].vector[i].subs(delta, dimensions[j]) * factor)
                    inner_list.append(extremal_entry)
                outer_list.append(inner_list)
            extremal_table.append(outer_list)

        # Determines the crossing equations where OPE coefficients only enter diagonally
        good_rows = []
        for i in range(0, len(self.unit)):
            j = 0
            good_row = True
            while j < zeros and good_row is True:
                size = len(extremal_table[j])
                for r in range(0, size):
                    for s in range(0, size):
                        if abs(extremal_table[j][r][s][i]) > tiny and r != s:
                            good_row = False
                j += 1
            if good_row is True:
                good_rows.append(i)

        fail = False
        known_ops = []
        # We go through the good rows, each time removing a chunk of them that uniformly include an OPE coefficient that is known
        # On the first iteration, when we do not know any, we pull out the ones that are inhomogeneous due to the identity
        while len(good_rows) > 0 and fail == False:
            other_rows = []
            current_rows = []
            current_coeffs = []
            new_dimensions = []
            new_spin_irreps = []

            current_target = [0, -1, -1]
            for i in good_rows:
                potential_coeffs = []
                if len(known_ops) == 0 and abs(self.unit[i]) < tiny:
                    other_rows.append(i)
                elif len(known_ops) == 0:
                    current_rows.append(i)
                elif current_target[0] == 0:
                    j = 0
                    found = False
                    while j < zeros and found is False:
                        size = len(extremal_table[j])
                        for vec in self.irrep_set:
                            if vec[1] == spin_irreps[j][1]:
                                break
                        r = 0
                        while r < size and found is False:
                            dim_set1 = [vec[0][0][r][r][0], vec[0][0][r][r][1], dimensions[j]]
                            dim_set1 = sorted(dim_set1)
                            for c in known_ops:
                                dim_set2 = [c[1], c[2], c[3]]
                                dim_set2 = sorted(dim_set2)
                                if abs(dim_set1[0] - dim_set2[0]) < 0.01 and abs(dim_set1[1] - dim_set2[1]) < 0.01 and abs(dim_set1[2] - dim_set2[2]) < 0.01:
                                    # OPE coefficient symmetry only holds with a particular normalization
                                    current_target = [(4.0 ** (dimensions[j] - c[3])) * c[0], j, r]
                                    found = True
                                    break
                            r += 1
                        j += 1
                    if found is False:
                        # This could happen if the SDP given to us does not correspond to the bootstrap of a physical theory
                        print("Leads exhausted")
                        fail = True
                if current_target[0] != 0:
                    j = current_target[1]
                    r = current_target[2]
                    if abs(extremal_table[j][r][r][i]) < tiny:
                        other_rows.append(i)
                    else:
                        current_rows.append(i)
            good_rows = other_rows

            # Determine all the OPE coefficients that could possibly be solved using these rows
            for i in current_rows:
                for j in range(0, zeros):
                    size = len(extremal_table[j])
                    for r in range(0, size):
                        if abs(extremal_table[j][r][r][i]) < tiny:
                            continue
                        if j == current_target[1] and r == current_target[2]:
                            continue
                        found_one = False
                        found_both = False
                        for c in current_coeffs:
                            if c[0] == j and c[1] == r:
                                found_one = True
                                found_both = True
                                break
                            elif c[0] == j:
                                found_one = True
                        if found_both is False:
                            current_coeffs.append((j, r))
                        if found_one is False:
                            new_dimensions.append(dimensions[j])
                            new_spin_irreps.append(spin_irreps[j])

            # If there are more operators than crossing equations, we must remove those of highest dimension
            if len(current_coeffs) + nullity > len(current_rows):
                refine = True
                kept_coeffs = []

                while refine is True:
                    index_new = new_dimensions.index(min(new_dimensions))
                    # Allow for different operators of the same dimension
                    target_dimension = new_dimensions[index_new]
                    target_spin_irrep = new_spin_irreps[index_new]
                    for index_old in range(0, len(dimensions)):
                        if abs(dimensions[index_old] - target_dimension) < tiny and spin_irreps[index_old] == target_spin_irrep:
                            break
                    new_coeffs = []
                    for pair in current_coeffs:
                        if pair[0] == index_old:
                            new_coeffs.append(pair)
                    if len(new_coeffs) + len(kept_coeffs) + nullity <= len(current_rows):
                        kept_coeffs = kept_coeffs + new_coeffs
                        new_dimensions = new_dimensions[:index_new] + new_dimensions[index_new + 1:]
                        new_spin_irreps = new_spin_irreps[:index_new] + new_spin_irreps[index_new + 1:]
                        refine = (len(new_dimensions) > 0)
                    else:
                        refine = False
                current_coeffs = kept_coeffs

            # If there are more crossing equations than operators, we must omit the ones corresponding to high derivatives
            # The last case might land us in this one as well if some OPE coefficients show up in pairs
            if len(current_rows) > len(current_coeffs) + nullity:
                current_rows = sorted(current_rows, key= lambda i: self.m_order[i] + self.n_order[i])
                current_rows = current_rows[:len(current_coeffs) + nullity]

            # Solve our system now that it is square
            identity = []
            extremal_blocks = []
            size = len(current_coeffs)
            if current_target[0] != 0:
                j_id = current_target[1]
                r_id = current_target[2]
            for i in current_rows:
                if current_target[0] == 0:
                    identity.append(self.unit[i])
                else:
                    identity.append(extremal_table[j_id][r_id][r_id][i])
                for pair in current_coeffs:
                    (j, r) = pair
                    extremal_blocks.append(float(extremal_table[j][r][r][i]))
            identity = DenseMatrix(size + nullity, 1, identity)
            extremal_matrix = DenseMatrix(size + nullity, size, extremal_blocks)
            if nullity == 0:
                solution = extremal_matrix.solve(identity)
            else:
                solution = self.least_absolute_distance(extremal_matrix, identity, aux_name=tmp_name)

            # Add these coefficients, along with other things we know, to the list of operators
            for i in range(0, len(current_coeffs)):
                (j, r) = current_coeffs[i]
                ope_coeff = solution.get(i, 0)
                for vec in self.irrep_set:
                    if vec[1] == spin_irreps[j][1]:
                        break
                dim1 = vec[0][r][r][0]
                dim2 = vec[0][r][r][1]
                known_ops.append([ope_coeff, dim1, dim2, dimensions[j], spin_irreps[j]])
        return known_ops

    def least_absolute_distance(self, matrix, vector, aux_name):
        """
        This returns the vector which is closest in the 1-norm to being a solution
        of an inhomogeneous linear system. It is convenient to overload `SDPB` as a
        linear program solver here.

        Parameters
        ----------
        matrix: A matrix having more rows than columns.
        vector: A vector whose length is the row dimension of `matrix`.
        aux_name: sdpDir for auxillary sdpb run
        """
        zeros = matrix.ncols()
        nullity = matrix.nrows() - zeros

        # The initial zero is the b_0 ignored by SDPB
        obj = [0]
        for i in range(0, zeros + nullity):
            obj.append(-1)
        for i in range(0, zeros):
            obj.append(0)

        constraint_vector = []
        for i in range(0, zeros + nullity):
            constraint_vector.append(vector.get(i, 0) * (-1))
        for i in range(0, zeros + nullity):
            constraint_vector.append(vector.get(i, 0))

        constraint_matrix = []
        for i in range(0, 2 * (zeros + nullity)):
            constraint_matrix.append([zero] * (2 * zeros + nullity))
        for i in range(0, zeros + nullity):
            constraint_matrix[i][i] = one
            constraint_matrix[zeros + nullity + i][i] = one
        for i in range(0, zeros + nullity):
            for j in range(0, zeros):
                constraint_matrix[i][zeros + nullity + j] = matrix.get(i, j) * (-1)
                constraint_matrix[zeros + nullity + i][zeros + nullity + j] = matrix.get(i, j)

        # To solve this with scipy, one could stop here
        # return linprog(-obj[1:], -constraint_matrix, -constraint_vector)

        extra = []
        for i in range(0, 2 * zeros + nullity):
            extra.append([zero] * (2 * zeros + nullity))
        for i in range(0, 2 * zeros + nullity):
            extra[i][i] = one
        constraint_matrix = extra + constraint_matrix
        constraint_vector = [zero] * (2 * zeros + nullity) + constraint_vector

        # Now that the functional components are positive, make a toy SDP for this
        aux_table1 = ConformalBlockTable(1, 0, 0, 0, 0)
        aux_table2 = ConvolvedBlockTable(aux_table1)
        aux_sdp = SDP(0, aux_table2, sdpb_mode=self.sdpb_mode, sdpb_kwargs=self.sdpb_kwargs)
        aux_sdp.sdpb.set_option("procsPerNode", self.sdpb.get_option("procsPerNode"))
        aux_sdp.bounds = [0] * len(constraint_vector)
        aux_sdp.basis = [DenseMatrix([[1]])] * len(constraint_vector)
        for i in range(0, len(constraint_vector)):
            block = [constraint_vector[i]] + constraint_matrix[i]
            aux_sdp.table.append([[PolynomialVector(block, [0, 0], [])]])

        norm = [0] * len(obj)
        norm[0] = -1

        aux_sdp.write_xml(obj, norm, name=aux_name)

        aux_sdp.sdpb.set_option("sdpDir", aux_name)
        aux_sdp.sdpb.set_option("procsPerNode", 1)
        aux_sdp.sdpb.set_option("noFinalCheckpoint", True)
        aux_sdp.sdpb.run({"noFinalCheckpoint": True})

        output = aux_sdp.sdpb.read_output(aux_sdp.sdpb.get_option("outDir"))
        solution = output["y"]
        solution = solution[zeros + nullity:]
        return DenseMatrix(zeros, 1, solution)
