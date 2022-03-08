from symengine.lib.symengine_wrapper import Symbol, RealMPFR, factorial

from .polynomial_vector import PolynomialVector
from .common import (
    ell, prec, delta, get_index_approx, index_iter, r_cross, omit_all,
    delta_ext, gather
)


class ConvolvedBlockTable:
    """
    A class which produces the functions that need to be linearly dependent in a
    crossing symmetric CFT. If a `ConformalBlockTable` does not need to be changed
    after a change to the external dimensions, a `ConvolvedBlockTable` does not
    either. This is because external dimensions only appear symbolically through a
    symbol called `delta_ext`.

    Parameters
    ----------
    block_table: A `ConformalBlockTable` from which to produce the convolved blocks.
    odd_spins:   [Optional] A parameter telling the class to keep odd spins which is
                 only used if `odd_spins` is True for `block_table`. Defaults to
                 `True`.
    symmetric:   [Optional] Whether to add blocks in two different channels instead
                 of subtract them. Defaults to `False`.
    content:     [Optional] A list of ordered triples that are used to produce
                 user-defined linear combinations of convolved conformal blocks
                 instead of just individual convolved conformal blocks where all the
                 coefficients are 1. Elements of a triple are taken to be the
                 coefficient, the dimension shift and the spin shift respectively.
                 It should always make sense to include a triple whose second and
                 third entries are 0 and 0 since this corresponds to a convolved
                 conformal block with scaling dimension `delta` and spin `ell`.
                 However, if other blocks in the multiplet have `delta + 1` and
                 `ell - 1` relative to this, another triple should be included whose
                 second and third entries are 1 and -1. The coefficient (first
                 entry) may be a polynomial in `delta` with coefficients depending
                 on `ell`.

    Attributes
    ----------
    dim:         The spatial dimension, inherited from `block_table`.
    k_max:       Numer controlling the accuracy of the rational approximation,
                 inherited from `block_table`.
    l_max:       The highest spin kept in the convolved block table. This is at most
                 the `l_max` of `block_table`.
    m_max:       Number controlling how many `a` derivatives there are where the
                 standard co-ordinates are expressed as `(a + sqrt(b)) / 2` and
                 `(a - sqrt(b)) / 2`. This is at most the `m_max` of `block_table`.
    n_max:       The number of `b` derivatives there are where the standard
                 co-ordinates are expressed as `(a + sqrt(b)) / 2` and
                 `(a - sqrt(b)) / 2`. This is at most the `n_max` of `block_table`.
    delta_12:    The difference between the external scaling dimensions of operator
                 1 and operator 2, inherited from `block_table`.
    delta_32:    The difference between the external scaling dimensions of operator
                 3 and operator 4, inherited from `block_table`.
    table:       A list of `PolynomialVector`s. A block's position in the table is
                 equal to its spin if `odd_spins` is `True`. Otherwise it is equal
                 to half of the spin.
    m_order:     A list stating how many `a` derivatives are being described by the
                 corresponding entry in a `PolynomialVector` in `table`. Different
                 from the `m_order` of `block_table` because some derivatives vanish
                 by symmetry.
    n_order:     A list stating how many `b` derivatives are being described by the
                 corresponding entry in a `PolynomialVector` in `table`.
    """
    def __init__(self, block_table, odd_spins = True, symmetric = False, spins = [], content = [[1, 0, 0]]):
        # Copying everything but the unconvolved table is fine from a memory standpoint
        self.dim = block_table.dim
        self.k_max = block_table.k_max
        self.l_max = block_table.l_max
        self.m_max = block_table.m_max
        self.n_max = block_table.n_max
        self.delta_12 = block_table.delta_12
        self.delta_34 = block_table.delta_34

        self.m_order = []
        self.n_order = []
        self.table = []

        max_spin_shift = 0
        for trip in content:
            max_spin_shift = max(max_spin_shift, trip[2])
        self.l_max -= max_spin_shift

        # We can restrict to even spin when the provided table has odd spin but not vice-versa
        if odd_spins == False and block_table.odd_spins == True:
            self.odd_spins = False
        else:
            self.odd_spins = block_table.odd_spins
        if block_table.odd_spins == True:
            step = 1
        else:
            step = 2
        if len(spins) > 0:
            spin_list = spins
        elif self.odd_spins:
            spin_list = range(0, self.l_max + 1, 1)
        else:
            spin_list = range(0, self.l_max + 1, 2)

        symbol_array = []
        for n in range(0, block_table.n_max + 1):
            symbol_list = []
            for m in range(0, 2 * (block_table.n_max - n) + block_table.m_max + 1):
                symbol_list.append(Symbol('g_' + n.__str__() + '_' + m.__str__()))
            symbol_array.append(symbol_list)

        derivatives = []
        for n in range(0, block_table.n_max + 1):
            for m in range(0, 2 * (block_table.n_max - n) + block_table.m_max + 1):
                # Skip the ones that will vanish
                if (symmetric == False and m % 2 == 0) or (symmetric == True and m % 2 == 1):
                    continue

                self.m_order.append(m)
                self.n_order.append(n)

                expression = 0
                old_coeff = RealMPFR("0.25", prec) ** delta_ext
                for j in range(0, n + 1):
                    coeff = old_coeff
                    for i in range(0, m + 1):
                        expression += coeff * symbol_array[n - j][m - i]
                        coeff *= (i + 2 * j - 2 * delta_ext) * (m - i) / (i + 1)
                    old_coeff *= (j - delta_ext) * (n - j) / (j + 1)

                deriv = expression / RealMPFR(str(factorial(m) * factorial(n)), prec)
                derivatives.append(deriv)

        combined_block_table = []
        for spin in spin_list:
            vector = []
            l = spin // step

            # Different blocks in the linear combination may be divided by different poles
            all_poles = []
            pole_dict = {}
            for trip in content:
                del_shift = trip[1]
                ell_shift = trip[2] // step
                if l + ell_shift >= 0:
                    gathered_poles = gather(block_table.table[l + ell_shift].poles)
                    for p in gathered_poles.keys():
                        ind = get_index_approx(pole_dict.keys(), p - del_shift)
                        if ind == -1:
                            pole_dict[p - del_shift] = gathered_poles[p]
                        else:
                            pole_dict_index = index_iter(pole_dict.keys(), ind)
                            num = pole_dict[pole_dict_index]
                            pole_dict[pole_dict_index] = max(num, gathered_poles[p])
            for p in pole_dict.keys():
                all_poles += [p] * pole_dict[p]

            for i in range(0, len(block_table.table[l].vector)):
                entry = 0
                for trip in content:
                    if "subs" in dir(trip[0]):
                        coeff = trip[0].subs(ell, spin)
                    else:
                        coeff = trip[0]
                    del_shift = trip[1]
                    ell_shift = trip[2] // step

                    coeff *= r_cross ** del_shift
                    if l + ell_shift >= 0:
                        coeff *= omit_all(all_poles, block_table.table[l + ell_shift].poles, delta, del_shift)
                        entry += coeff * block_table.table[l + ell_shift].vector[i].subs(delta, delta + del_shift)
                vector.append(entry.expand())
            combined_block_table.append(PolynomialVector(vector, [spin, 0], all_poles))

        for l in range(0, len(combined_block_table)):
            new_derivs = []
            for i in range(0, len(derivatives)):
                deriv = derivatives[i]
                for j in range(len(combined_block_table[l].vector) - 1, 0, -1):
                    deriv = deriv.subs(symbol_array[block_table.n_order[j]][block_table.m_order[j]], combined_block_table[l].vector[j])
                new_derivs.append(2 * deriv.subs(symbol_array[0][0], combined_block_table[l].vector[0]))
            self.table.append(PolynomialVector(new_derivs, combined_block_table[l].label, combined_block_table[l].poles))
