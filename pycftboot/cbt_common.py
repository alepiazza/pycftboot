from abc import ABC, abstractmethod

from .common import get_index_approx
from .constants import delta


class ConformalBlockTableCommon(ABC):
    """
    A meta class which calculates tables of conformal block derivatives from scratch

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

        if compute:
            self.compute_table()

    def __eq__(self, other):
        if not isinstance(other, ConformalBlockTableCommon):
            # don't attempt to compare against unrelated types
            return NotImplemented
        return self.__dict__ == other.__dict__

    def compute_table(self):
        self.m_order, self.n_order, self.table = self._compute_table(
            self.dim, self.k_max, self.l_max, self.m_max, self.n_max, self.delta_12, self.delta_34, self.odd_spins
        )

    @abstractmethod
    def _compute_table(self, dim, k_max, l_max, m_max, n_max, delta_12, delta_34, odd_spins):
        pass

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
        if not isinstance(tab_long, ConformalBlockTableCommon):
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
