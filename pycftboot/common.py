import itertools
from symengine.lib.symengine_wrapper import (
    RealMPFR, Symbol, sqrt, function_symbol, Derivative, Subs, Integer
)

from .constants import tiny, prec, delta


def rf(x, n):
    """
    Implements the rising factorial or Pochhammer symbol.
    """
    ret = 1
    if n < 0:
        return rf(x - abs(n), abs(n)) ** (-1)
    for k in range(0, n):
        ret *= x + k
    return ret


def deepcopy(array):
    """
    Copies a list of a list so that entries can be changed non-destructively.
    """
    ret = []
    for el in array:
        ret.append(list(el))
    return ret


def index_iter(iter, n):
    """
    Returns the nth element of an iterator.
    """
    return next(itertools.islice(iter, n, None))


def get_index(array, element, start=0):
    """
    Finds where an element occurs in an array or -1 if not present.
    """
    for i, v in itertools.islice(enumerate(array), start, None):
        if v == element:
            return i
    return -1


def get_index_approx(array, element, start=0):
    """
    Finds where an element numerically close to the one given occurs in an array
    or -1 if not present.
    """
    for i, v in itertools.islice(enumerate(array), start, None):
        if abs(v - element) < tiny:
            return i
    return -1


def gather(array):
    """
    Finds (approximate) duplicates in a list and returns a dictionary that counts
    the number of appearances.
    """
    ret = {}
    backup = list(array)
    while len(backup) > 0:
        i = 0
        hits = []
        while i >= 0:
            hits.append(i)
            i = get_index_approx(backup, backup[i], i + 1)
        ret[backup[0]] = len(hits)
        hits.reverse()
        for i in hits:
            backup = backup[:i] + backup[i + 1:]
    return ret


def extract_power(term):
    """
    Returns the degree of a single term in a polynomial. Symengine stores these
    as (coefficient, (delta, exponent)). This is helpful for sorting polynomials
    which are not sorted by default.
    """
    if "args" not in dir(term):
        return 0

    if term.args == ():
        return 0
    elif term.args[1].args == ():
        return 1
    else:
        return int(term.args[1].args[1])


def coefficients(polynomial):
    """
    Returns a sorted list of all coefficients in a polynomial starting with the
    constant term. Zeros are automatically added so that the length of the list
    is always one more than the degree.
    """
    if "args" not in dir(polynomial):
        return [polynomial]
    if polynomial.args == ():
        return [polynomial]

    coeff_list = sorted(polynomial.args, key=extract_power)
    degree = extract_power(coeff_list[-1])

    pos = 0
    ret = []
    for d in range(0, degree + 1):
        if extract_power(coeff_list[pos]) == d:
            if d == 0:
                ret.append(RealMPFR(str(coeff_list[0]), prec))
            else:
                ret.append(RealMPFR(str(coeff_list[pos].args[0]), prec))
            pos += 1
        else:
            ret.append(0)
    return ret


def build_polynomial(coefficients):
    """
    Returns a polynomial in `delta` from a list of coefficients. The first one is
    expected to be the constant term.
    """
    ret = 0
    prod = 1
    for d in range(0, len(coefficients)):
        ret += coefficients[d] * prod
        prod *= delta
    return ret


def unitarity_bound(dim, spin):
    """
    Returns the lower bound for conformal dimensions in a unitary theory for a
    given spatial dimension and spin.
    """
    if spin == 0:
        return (dim / Integer(2)) - 1
    else:
        return dim + spin - 2


def omit_all(poles, special_poles, var, shift=0):
    """
    Instead of returning a product of poles where each pole is not in a special
    list, this returns a product where each pole is subtracted from some variable.
    """
    expression = 1
    gathered1 = gather(poles)
    gathered0 = gather(special_poles)
    for p in gathered1.keys():
        ind = get_index_approx(gathered0.keys(), p + shift)
        if ind == -1:
            power = 0
        else:
            power = gathered0[index_iter(gathered0.keys(), ind)]
        expression *= (var - p) ** (gathered1[p] - power)
    return expression


def rules(m_max, n_max):
    """
    This takes the radial and angular co-ordinates, defined by Hogervorst and
    Rychkov in arXiv:1303.1111, and differentiates them with respect to the
    diagonal `a` and off-diagonal `b`. It returns a quadruple where the first
    two entries store radial and angular derivatives respectively evaluated at
    the crossing symmetric point. The third entry is a list stating the number of
    `a` derivatives to which a given position corresponds and the fourth entry
    does the same for `b` derivatives.
    """
    a = Symbol('a')
    b = Symbol('b')
    hack = Symbol('hack')

    rules1 = []
    rules2 = []
    m_order = []
    n_order = []
    old_expression1 = sqrt(a ** 2 - b) / (hack + sqrt((hack - a) ** 2 - b) + hack * sqrt(hack - a + sqrt((hack - a) ** 2 - b)))
    old_expression2 = (hack - sqrt((hack - a) ** 2 - b)) / sqrt(a ** 2 - b)

    if n_max == 0:
        old_expression1 = old_expression1.subs(b, 0)
        old_expression2 = b

    for n in range(0, n_max + 1):
        for m in range(0, 2 * (n_max - n) + m_max + 1):
            if n == 0 and m == 0:
                expression1 = old_expression1
                expression2 = old_expression2
            elif m == 0:
                old_expression1 = old_expression1.diff(b)
                old_expression2 = old_expression2.diff(b)
                expression1 = old_expression1
                expression2 = old_expression2
            else:
                expression1 = expression1.diff(a)
                expression2 = expression2.diff(a)

            rules1.append(expression1.subs({hack: RealMPFR("2", prec), a: 1, b: 0}))
            rules2.append(expression2.subs({hack: RealMPFR("2", prec), a: 1, b: 0}))
            m_order.append(m)
            n_order.append(n)

    return (rules1, rules2, m_order, n_order)


def delta_pole(nu: RealMPFR, k, l, series):
    """
    Returns the pole of a meromorphic global conformal block given by the
    parameters in arXiv:1406.4858 by Kos, Poland and Simmons-Duffin.

    Parameters
    ----------
    nu:     `(d - 2) / 2` where d is the spatial dimension.
    k:      The parameter k indexing the various poles. As described in
            arXiv:1406.4858, it may be any positive integer unless `series` is 3.
    l:      The spin.
    series: The parameter i desribing the three types of poles in arXiv:1406.4858.
    """
    if series == 1:
        pole = 1 - l - k
    elif series == 2:
        pole = 1 + nu - k
    else:
        pole = 1 + l + 2 * nu - k

    return pole.evalf(prec)
