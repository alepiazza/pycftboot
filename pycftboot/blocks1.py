from symengine.lib.symengine_wrapper import factorial, Integer, DenseMatrix

from .cbt_common import ConformalBlockTableCommon
from .polynomial_vector import PolynomialVector
from .block_vector import (
    LeadingBlockVector, MeromorphicBlockVector, ConformalBlockVector
)
from .chain_rule import chain_rule_single, chain_rule_double
from .common import rf, rules, delta_pole
from .constants import prec, two, r_cross, delta


class ConformalBlockTableSeed1(ConformalBlockTableCommon):
    """
    A class which calculates tables of conformal block derivatives from scratch
    using the recursion relations with meromorphic versions of the blocks.
    Usually, it will not be necessary for the user to call it. Instead,
    `ConformalBlockTable` calls it automatically for `m_max = 3` and `n_max = 0`.
    For people wanting to call it with different values of `m_max` and `n_max`,
    the parameters and attributes are the same as those of `ConformalBlockTable`.
    It also supports the `dump` method.
    """

    def _compute_table(self, dim, k_max, l_max, m_max, n_max, delta_12, delta_34, odd_spins):
        m_order = []
        n_order = []
        table = []

        if odd_spins:
            step = 1
        else:
            step = 2

        derivative_order = m_max + 2 * n_max
        nu = (dim / Integer(2)) - 1

        # The matrix for how derivatives are affected when one multiplies by r
        r_powers = []
        identity = [0] * ((derivative_order + 1) ** 2)
        lower_band = [0] * ((derivative_order + 1) ** 2)

        for i in range(0, derivative_order + 1):
            identity[i * (derivative_order + 1) + i] = 1
        for i in range(1, derivative_order + 1):
            lower_band[i * (derivative_order + 1) + i - 1] = i

        identity = DenseMatrix(derivative_order + 1, derivative_order + 1, identity)
        lower_band = DenseMatrix(derivative_order + 1, derivative_order + 1, lower_band)
        r_matrix = identity.mul_scalar(r_cross).add_matrix(lower_band)
        r_powers.append(identity)
        r_powers.append(r_matrix)

        conformal_blocks = []
        leading_blocks = []
        pol_list = []
        res_list = []
        pow_list = []
        new_res_list = []
        new_pow_list = []

        # Find out which residues we will ever need to include
        for l in range(0, l_max + k_max + 1):
            lb = LeadingBlockVector(dim, l, m_max, n_max, delta_12, delta_34)
            leading_blocks.append(lb)
            current_pol_list = []

            for k in range(1, k_max + 1):
                if l + k <= l_max + k_max:
                    if delta_residue(nu, k, l, delta_12, delta_34, 1) != 0:
                        current_pol_list.append((k, k, l + k, 1))

                if k % 2 == 0:
                    if delta_residue(nu, k // 2, l, delta_12, delta_34, 2) != 0:
                        current_pol_list.append((k, k // 2, l, 2))

                if k <= l:
                    if delta_residue(nu, k, l, delta_12, delta_34, 3) != 0:
                        current_pol_list.append((k, k, l - k, 3))

                if l == 0:
                    r_powers.append(r_powers[k].mul_matrix(r_powers[1]))

            # These are in the format (n, k, l, series)
            pol_list.append(current_pol_list)
            res_list.append([])
            pow_list.append([])
            new_res_list.append([])
            new_pow_list.append([])

        # Initialize the residues at the appropriate leading blocks
        for l in range(0, l_max + k_max + 1):
            for i in range(0, len(pol_list[l])):
                l_new = pol_list[l][i][2]
                res_list[l].append(MeromorphicBlockVector(leading_blocks[l_new]))
                pow_list[l].append(0)

                new_pow_list[l].append(pol_list[l][i][0])
                new_res_list[l].append(0)

        for k in range(1, k_max + 1):
            for l in range(0, l_max + k_max + 1):
                for i in range(0, len(res_list[l])):
                    if pow_list[l][i] >= k_max:
                        continue

                    res = delta_residue(nu, pol_list[l][i][1], l, delta_12, delta_34, pol_list[l][i][3])
                    pow_list[l][i] = new_pow_list[l][i]

                    for j in range(0, len(res_list[l][i].chunks)):
                        r_sub = r_powers[pol_list[l][i][0]][0:derivative_order - j + 1, 0:derivative_order - j + 1]
                        res_list[l][i].chunks[j] = r_sub.mul_matrix(res_list[l][i].chunks[j]).mul_scalar(res)

            for l in range(0, l_max + k_max + 1):
                for i in range(0, len(res_list[l])):
                    if pow_list[l][i] >= k_max:
                        continue

                    new_pow = k_max
                    l_new = pol_list[l][i][2]
                    new_res_list[l][i] = MeromorphicBlockVector(leading_blocks[l_new])
                    pole1 = delta_pole(nu, pol_list[l][i][1], l, pol_list[l][i][3]) + pol_list[l][i][0]

                    for i_new in range(0, len(res_list[l_new])):
                        new_pow = min(new_pow, pol_list[l_new][i_new][0])
                        pole2 = delta_pole(nu, pol_list[l_new][i_new][1], l_new, pol_list[l_new][i_new][3])

                        for j in range(0, len(new_res_list[l][i].chunks)):
                            new_res_list[l][i].chunks[j] = new_res_list[l][i].chunks[j].add_matrix(res_list[l_new][i_new].chunks[j].mul_scalar(1 /(pole1 - pole2).evalf(prec)))

                    new_pow_list[l][i] = pow_list[l][i] + new_pow

            for l in range(0, l_max + k_max + 1):
                for i in range(0, len(res_list[l])):
                    if pow_list[l][i] >= k_max:
                        continue

                    for j in range(0, len(res_list[l][i].chunks)):
                        res_list[l][i].chunks[j] = new_res_list[l][i].chunks[j]

        # Perhaps poorly named, S keeps track of a linear combination of derivatives
        # We get this by including the essential singularity, then stripping it off again
        s_matrix = DenseMatrix(derivative_order + 1, derivative_order + 1, [0] * ((derivative_order + 1) ** 2))
        for i in range(0, derivative_order + 1):
            new_element = 1
            for j in range(i, -1, -1):
                s_matrix.set(i, j, new_element)
                new_element *= (j / ((i - j + 1) * r_cross)) * (delta - (i - j))

        for l in range(0, l_max + 1, step):
            conformal_block = ConformalBlockVector(dim, l, delta_12, delta_34, m_max + 2 * n_max, k_max, s_matrix, leading_blocks[l], pol_list[l], res_list[l])
            conformal_blocks.append(conformal_block)
            table.append(PolynomialVector([], [l, 0], conformal_block.large_poles))

        (rules1, rules2, m_order, n_order) = rules(m_max, n_max)
        # If b is always 0, then eta is always 1
        if n_max == 0:
            chain_rule_single(m_order, rules1, table, conformal_blocks, lambda l, i: conformal_blocks[l].chunks[0].get(i, 0))
        else:
            chain_rule_double(m_order, n_order, rules1, rules2, table, conformal_blocks)

        return (m_order, n_order, table)

    # def dump(self, name, form=None):
    #     if form == "juliboots":
    #         juliboots_write(self, name)
    #     elif form == "scalar_blocks":
    #         scalar_blocks_write(self, name)
    #     else:
    #         dump_table_contents(self, name)


def delta_residue(nu, k, l, delta_12, delta_34, series):
    """
    Returns the residue of a meromorphic global conformal block at a particular
    pole in `delta`. These residues were found by Kos, Poland and Simmons-Duffin
    in arXiv:1406.4858.

    Parameters
    ----------
    nu:       `(d - 2) / 2` where d is the spatial dimension. This must be
              different from an integer.
    k:        The parameter k indexing the various poles. As described in
              arXiv:1406.4858, it may be any positive integer unless `series`
              is 3.
    l:        The spin.
    delta_12: The difference between the external scaling dimensions of operator
              1 and operator 2.
    delta_34: The difference between the external scaling dimensions of operator
              3 and operator 4.
    series:   The parameter i desribing the three types of poles in
              arXiv:1406.4858.
    """
    # Time saving special case
    if series != 2 and k % 2 != 0 and delta_12 == 0 and delta_34 == 0:
        return 0

    if series == 1:
        ret = - ((k * (-4) ** k) / (factorial(k) ** 2)) * rf((1 - k + delta_12) / two, k) * rf((1 - k + delta_34) / two, k)
        if ret == 0:
            return ret
        elif l == 0 and nu == 0:
            # Take l to 0, then nu
            return ret * 2
        else:
            return ret * (rf(l + 2 * nu, k) / rf(l + nu, k))
    elif series == 2:
        factors = [l + nu + 1 - delta_12, l + nu + 1 + delta_12, l + nu + 1 - delta_34, l + nu + 1 + delta_34]
        ret = ((k * rf(nu + 1, k - 1)) / (factorial(k) ** 2)) * ((l + nu - k) / (l + nu + k))
        ret *= rf(-nu, k + 1) / ((rf((l + nu - k + 1) / 2, k) * rf((l + nu - k) / 2, k)) ** 2)

        for f in factors:
            ret *= rf((f - k) / two, k)
        return ret
    else:
        return - ((k * (-4) ** k) / (factorial(k) ** 2)) * (rf(1 + l - k, k) * rf((1 - k + delta_12) / two, k) * rf((1 - k + delta_34) / two, k) / rf(1 + nu + l - k, k))
