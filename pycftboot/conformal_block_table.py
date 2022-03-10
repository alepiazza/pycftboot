from symengine.lib.symengine_wrapper import Symbol, RealMPFR

from .blocks1 import ConformalBlockTableSeed
from .blocks2 import ConformalBlockTableSeed2
from .common import get_index_approx
from .constants import ell, prec, delta, two, one


class ConformalBlockTable:
    """
    A class which calculates tables of conformal block derivatives when initialized.
    This uses recursion relations on the diagonal found by Hogervorst, Osborn and
    Rychkov in arXiv:1305.1321.

    Parameters
    ----------
    dim:       The spatial dimension. If even dimensions are of interest, floating
               point numbers with small fractional parts are recommended.
    k_max:     Number controlling the accuracy of the rational approximation.
               Specifically, it is the maximum power of the crossing symmetric value
               of the radial co-ordinate as described in arXiv:1406.4858.
    l_max:     The maximum spin to include in the table.
    m_max:     Number controlling how many `a` derivatives to include where the
               standard co-ordinates are expressed as `(a + sqrt(b)) / 2` and
               `(a - sqrt(b)) / 2`. As explained in arXiv:1412.4127, a value of 0
               does not necessarily eliminate all `a` derivatives.
    n_max:     The number of `b` derivatives to include where the standard
               co-ordinates are expressed as `(a + sqrt(b)) / 2` and
               `(a - sqrt(b)) / 2`.
    delta_12:  [Optional] The difference between the external scaling dimensions of
               operator 1 and operator 2. Defaults to 0.
    delta_34:  [Optional] The difference between the external scaling dimensions of
               operator 3 and operator 4. Defaults to 0.
    odd_spins: [Optional] Whether to include 0, 1, 2, 3, ..., `l_max` instead of
               just 0, 2, 4, ..., `l_max`. Defaults to `False`.
    name:      [Optional] The name of a file containing conformal blocks that have
               already been calculated. If this is specified and refers to a file
               produced by PyCFTBoot, all other parameters passed to the class are
               overwritten by the ones in the table. If it refers to one produced
               by JuliBoots, all parameters passed to the class except delta_12 and
               delta_34 are overwritten. If it refers to a directory, the contents
               will be treated as scalar_blocks output files. Parameters will be
               only be overwritten in this case if they can easily be parsed from
               the filenames.

    Attributes
    ----------
    table:     A list of `PolynomialVector`s. A block's position in the table is
               equal to its spin if `odd_spins` is True. Otherwise it is equal to
               half of the spin.
    m_order:   A list with the same number of components as the `PolynomialVector`s
               in `table`. Any `i`-th entry in a `PolynomialVector` is a particular
               derivative of a conformal block, but to remember which one, just look
               at the `i`-th entry of `m_order` which is the number of `a`
               derivatives.
    n_order:   A list with the same number of components as the `PolynomialVector`s
               in `table`. Any `i`-th entry in a `PolynomialVector` is a particular
               derivative of a conformal block, but to remember which one, just look
               at the `i`-th entry of `n_order` which is the number of `b`
               derivatives.
    """

    def __init__(self, dim, k_max, l_max, m_max, n_max, delta_12=0, delta_34=0, odd_spins=False, compute=True):
        self.dim = dim
        self.k_max = k_max
        self.l_max = l_max
        self.m_max = m_max
        self.n_max = n_max
        self.delta_12 = delta_12
        self.delta_34 = delta_34
        self.odd_spins = odd_spins
        self.m_order = []
        self.n_order = []
        self.table = []

        if compute is True:
            self.compute_table()

    def __eq__(self, other):
        if not isinstance(other, ConformalBlockTable):
            # don't attempt to compare against unrelated types
            return NotImplemented
        return self.__dict__ == other.__dict__

    def compute_table(self):
        self.n_order, self.m_order, self.table = compute_conformal_block_table(
            self.dim, self.k_max, self.l_max, self.m_max, self.n_max, self.delta_12, self.delta_34, self.odd_spins
        )

    def convert_table(self, tab_long):
        """
        Converts the table attribute having few poles into an equivalent table with many poles.
        When tables produced by different methods fail to look the same, it is often
        because their polynomials are being multiplied by different positive
        prefactors. This adjusts the prefactors so that they are the same.

        Parameters
        ----------
        tab_long:  A `ConformalBlockTable` with all of the poles that `self.table` has
                   plus more.
        """
        if not isinstance(tab_long, ConformalBlockTable):
            raise TypeError(f"{tab_long} is not a ConformalBlockTable")

        for l in range(0, len(self.table)):
            pole_prod = 1
            small_list = self.table[l].poles[:]

            for p in tab_long.table[l].poles:
                index = get_index_approx(small_list, p)

                if index == -1:
                    pole_prod *= delta - p
                    self.table[l].poles.append(p)
                else:
                    small_list.remove(small_list[index])

            for n in range(0, len(self.table[l].vector)):
                self.table[l].vector[n] = self.table[l].vector[n] * pole_prod
                self.table[l].vector[n] = self.table[l].vector[n].expand()


def compute_conformal_block_table(dim, k_max, l_max, m_max, n_max, delta_12, delta_34, odd_spins):
    if isinstance(dim, int) and dim % 2 == 0:
        small_table = ConformalBlockTableSeed2(dim, k_max, l_max, min(m_max + 2 * n_max, 3), delta_12, delta_34, odd_spins)
    else:
        small_table = ConformalBlockTableSeed(dim, k_max, l_max, min(m_max + 2 * n_max, 3), 0, delta_12, delta_34, odd_spins)

    m_order = small_table.m_order
    n_order = small_table.n_order
    table = small_table.table

    a = Symbol('a')
    nu = RealMPFR(str(dim - 2), prec) / 2
    c_2 = (ell * (ell + 2 * nu) + delta * (delta - 2 * nu - 2)) / 2
    c_4 = ell * (ell + 2 * nu) * (delta - 1) * (delta - 2 * nu - 1)
    polys = [0, 0, 0, 0, 0]
    poly_derivs = [[], [], [], [], []]
    delta_prod = -delta_12 * delta_34 / two
    delta_sum = -(delta_12 - delta_34) / two

    # Polynomial 0 goes with the lowest order derivative on the right hand side
    # Polynomial 3 goes with the highest order derivative on the right hand side
    # Polynomial 4 goes with the derivative for which we are solving
    polys[0] += (a ** 0) * (16 * c_2 * (2 * nu + 1) - 8 * c_4)
    polys[0] += (a ** 1) * (4 * (c_4 + 2 * (2 * nu + 1) * (c_2 * delta_sum - c_2 + nu * delta_prod)))
    polys[0] += (a ** 2) * (2 * (delta_sum - nu) * (c_2 * (2 * delta_sum - 1) + delta_prod * (6 * nu - 1)))
    polys[0] += (a ** 3) * (2 * delta_prod * (delta_sum - nu) * (delta_sum - nu + 1))
    polys[1] += (a ** 1) * (-16 * c_2 * (2 * nu + 1))
    polys[1] += (a ** 2) * (4 * delta_prod - 24 * nu * delta_prod + 8 * nu * (2 * nu - 1) * (2 * delta_sum + 1) + 4 * c_2 * (1 - 4 * delta_sum + 6 * nu))
    polys[1] += (a ** 3) * (2 * c_2 * (4 * delta_sum - 2 * nu + 1) + 4 * (2 * nu - 1) * (2 * delta_sum + 1) * (delta_sum - nu + 1) + 2 * delta_prod * (10 * nu - 5 - 4 * delta_sum))
    polys[1] += (a ** 4) * ((delta_sum - nu + 1) * (4 * delta_prod + (2 * delta_sum + 1) * (delta_sum - nu + 2)))
    polys[2] += (a ** 2) * (16 * c_2 + 16 * nu - 32 * nu * nu)
    polys[2] += (a ** 3) * (8 * delta_prod - 8 * (3 * delta_sum - nu + 3) * (2 * nu - 1) - 16 * c_2 - 8 * nu + 16 * nu * nu)
    polys[2] += (a ** 4) * (4 * (c_2 - delta_prod + (3 * delta_sum - nu + 3) * (2 * nu - 1)) - 4 * delta_prod - 2 * (delta_sum - nu + 2) * (5 * delta_sum - nu + 5))
    polys[2] += (a ** 5) * (2 * delta_prod + (delta_sum - nu + 2) * (5 * delta_sum - nu + 5))
    polys[3] += (a ** 3) * (32 * nu - 16)
    polys[3] += (a ** 4) * (16 - 32 * nu + 4 * (4 * delta_sum - 2 * nu + 7))
    polys[3] += (a ** 5) * (4 * (2 * nu - 1) - 4 * (4 * delta_sum - 2 * nu + 7))
    polys[3] += (a ** 6) * (4 * delta_sum - 2 * nu + 7)
    polys[4] += (a ** 7) - 6 * (a ** 6) + 12 * (a ** 5) - 8 * (a ** 4)

    # Store all possible derivatives of these polynomials
    for i in range(0, 5):
        for j in range(0, i + 4):
            poly_derivs[i].append(polys[i].subs(a, 1))
            polys[i] = polys[i].diff(a)

    for m in range(m_order[-1] + 1, m_max + 2 * n_max + 1):
        for l in range(0, len(small_table.table)):
            new_deriv = 0
            for i in range(m - 1, max(m - 8, -1), -1):
                coeff = 0
                index = max(m - i - 4, 0)

                prefactor = one
                for k in range(0, index):
                    prefactor *= (m - 4 - k)
                    prefactor /= k + 1

                k = max(4 + i - m, 0)
                while k <= 4 and index <= (m - 4):
                    coeff += prefactor * poly_derivs[k][index]
                    prefactor *= (m - 4 - index)
                    prefactor /= index + 1
                    index += 1
                    k += 1

                if not isinstance(coeff, int):
                    coeff = coeff.subs(ell, small_table.table[l].label[0])
                new_deriv -= coeff * table[l].vector[i]

            new_deriv = new_deriv / poly_derivs[4][0]
            table[l].vector.append(new_deriv.expand())

        m_order.append(m)
        n_order.append(0)

    # This is just an alternative to storing derivatives as a doubly-indexed list
    index = m_max + 2 * n_max + 1
    index_map = [range(0, m_max + 2 * n_max + 1)]

    for n in range(1, n_max + 1):
        index_map.append([])
        for m in range(0, 2 * (n_max - n) + m_max + 1):
            index_map[n].append(index)

            coeff1 = m * (-1) * (2 - 4 * n - 4 * nu)
            coeff2 = m * (m - 1) * (2 - 4 * n - 4 * nu)
            coeff3 = m * (m - 1) * (m - 2) * (2 - 4 * n - 4 * nu)
            coeff4 = 1
            coeff5 = (-6 + m + 4 * n - 2 * nu - 2 * delta_sum)
            coeff6 = (-1) * (4 * c_2 + m * m + 8 * m * n - 5 * m + 4 * n * n - 2 * n - 2 - 4 * nu * (1 - m - n) + 4 * delta_sum * (m + 2 * n - 2) + 2 * delta_prod)
            coeff7 = m * (-1) * (m * m + 12 * m * n - 13 * m + 12 * n * n - 34 * n + 22 - 2 * nu * (2 * n - m - 1) + 2 * delta_sum * (m + 4 * n - 5) + 2 * delta_prod)
            coeff8 = (1 - n)
            coeff9 = (1 - n) * (-6 + 3 * m + 4 * n - 2 * nu + 2 * delta_sum)

            for l in range(0, len(small_table.table)):
                new_deriv = 0

                if m > 0:
                    new_deriv += coeff1 * table[l].vector[index_map[n][m - 1]]
                if m > 1:
                    new_deriv += coeff2 * table[l].vector[index_map[n][m - 2]]
                if m > 2:
                    new_deriv += coeff3 * table[l].vector[index_map[n][m - 3]]

                new_deriv += coeff4 * table[l].vector[index_map[n - 1][m + 2]]
                new_deriv += coeff5 * table[l].vector[index_map[n - 1][m + 1]]
                new_deriv += coeff6.subs(ell, small_table.table[l].label[0]) * table[l].vector[index_map[n - 1][m]]
                new_deriv += coeff7 * table[l].vector[index_map[n - 1][m - 1]]

                if n > 1:
                    new_deriv += coeff8 * table[l].vector[index_map[n - 2][m + 2]]
                    new_deriv += coeff9 * table[l].vector[index_map[n - 2][m + 1]]

                new_deriv = new_deriv / (2 - 4 * n - 4 * nu)
                table[l].vector.append(new_deriv.expand())

            m_order.append(m)
            n_order.append(n)
            index += 1

    return (m_order, n_order, table)
