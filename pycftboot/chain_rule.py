from symengine.lib.symengine_wrapper import function_symbol, Symbol, Derivative, Subs

from .common import deepcopy


def chain_rule_single_symengine(m_order, rules, table, conformal_blocks, accessor):
    """
    This reads a conformal block list where each spin's entry is a list of radial
    derivatives. It converts these to diagonal `a` derivatives using the rules
    given. Once these are calculated, the passed `table` is populated in place. Here,
    `accessor` is a hack to get around the fact that different parts of the code
    like to index in different ways.
    """
    _x = Symbol('_x')
    a = Symbol('a')
    r = function_symbol('r', a)
    g = function_symbol('g', r)
    m_max = max(m_order)

    for m in range(0, m_max + 1):
        if m == 0:
            old_expression = g
            g = function_symbol('g', _x)
        else:
            old_expression = old_expression.diff(a)

        expression = old_expression
        for i in range(1, m + 1):
            expression = expression.subs(Derivative(r, [a] * m_order[i]), rules[i])

        for l in range(0, len(conformal_blocks)):
            new_deriv = expression
            for i in range(1, m + 1):
                new_deriv = new_deriv.subs(Subs(Derivative(g, [_x] * i), [_x], [r]), accessor(l, i))
            if m == 0:
                new_deriv = accessor(l, 0)
            table[l].vector.append(new_deriv.expand())


def chain_rule_single(m_order, rules, table, conformal_blocks, accessor):
    """
    This implements the same thing except in Python which should not be faster
    but it is.
    """
    a = Symbol('a')
    r = function_symbol('r', a)
    m_max = max(m_order)

    old_coeff_grid = [0] * (m_max + 1)
    old_coeff_grid[0] = 1
    order = 0

    for m in range(0, m_max + 1):
        if m == 0:
            coeff_grid = old_coeff_grid[:]
        else:
            for i in range(m - 1, -1, -1):
                coeff = coeff_grid[i]
                if type(coeff) == type(1):
                    coeff_deriv = 0
                else:
                    coeff_deriv = coeff.diff(a)
                coeff_grid[i + 1] += coeff * r.diff(a)
                coeff_grid[i] = coeff_deriv

        deriv = coeff_grid[:]
        for l in range(order, 0, -1):
            for i in range(0, m + 1):
                if type(deriv[i]) != type(1):
                    deriv[i] = deriv[i].subs(Derivative(r, [a] * m_order[l]), rules[l])

        for l in range(0, len(conformal_blocks)):
            new_deriv = 0
            for i in range(0, m + 1):
                new_deriv += deriv[i] * accessor(l, i)
            table[l].vector.append(new_deriv.expand())
        order += 1


def chain_rule_double_symengine(m_order, n_order, rules1, rules2, table, conformal_blocks):
    """
    This reads a conformal block list where each spin has a chunk for a given
    number of angular derivatives and different radial derivatives within each
    chunk. It converts these to diagonal and off-diagonal `a` and `b` derivatives
    using the two sets of rules given. Once these are calculated, the passed
    `table` is populated in place.
    """
    _x = Symbol('_x')
    __x = Symbol('__x')
    a = Symbol('a')
    b = Symbol('b')
    r = function_symbol('r', a, b)
    eta = function_symbol('eta', a, b)
    g = function_symbol('g', r, eta)
    n_max = max(n_order)
    m_max = max(m_order) - 2 * n_max
    order = 0

    for n in range(0, n_max + 1):
        for m in range(0, 2 * (n_max - n) + m_max + 1):
            if n == 0 and m == 0:
                old_expression = g
                expression = old_expression
                g0 = function_symbol('g', __x, _x)
                g1 = function_symbol('g', _x, __x)
                g2 = function_symbol('g', _x, eta)
                g3 = function_symbol('g', r, _x)
                g4 = function_symbol('g', r, eta)
            elif m == 0:
                old_expression = old_expression.diff(b)
                expression = old_expression
            else:
                expression = expression.diff(a)

            deriv = expression
            for l in range(order, 0, -1):
                deriv = deriv.subs(Derivative(r, [a] * m_order[l] + [b] * n_order[l]), rules1[l])
                deriv = deriv.subs(Derivative(r, [b] * n_order[l] + [a] * m_order[l]), rules1[l])
                deriv = deriv.subs(Derivative(eta, [a] * m_order[l] + [b] * n_order[l]), rules2[l])
                deriv = deriv.subs(Derivative(eta, [b] * n_order[l] + [a] * m_order[l]), rules2[l])

            for l in range(0, len(conformal_blocks)):
                new_deriv = deriv
                for i in range(1, m + n + 1):
                    for j in range(1, m + n - i + 1):
                        new_deriv = new_deriv.subs(Subs(Derivative(g1, [_x] * i + [__x] * j), [_x, __x], [r, eta]), conformal_blocks[l].chunks[j].get(i, 0))
                        new_deriv = new_deriv.subs(Subs(Derivative(g0, [_x] * j + [__x] * i), [_x, __x], [eta, r]), conformal_blocks[l].chunks[j].get(i, 0))
                for i in range(1, m + n + 1):
                    new_deriv = new_deriv.subs(Subs(Derivative(g2, [_x] * i), [_x], [r]), conformal_blocks[l].chunks[0].get(i, 0))
                for j in range(1, m + n + 1):
                    new_deriv = new_deriv.subs(Subs(Derivative(g3, [_x] * j), [_x], [eta]), conformal_blocks[l].chunks[j].get(0, 0))
                new_deriv = new_deriv.subs(g4, conformal_blocks[l].chunks[0].get(0, 0))
                table[l].vector.append(new_deriv.expand())
            order += 1


def chain_rule_double(m_order, n_order, rules1, rules2, table, conformal_blocks):
    """
    This implements the same thing except in Python which should not be faster
    but it is.
    """
    a = Symbol('a')
    b = Symbol('b')
    r = function_symbol('r', a, b)
    eta = function_symbol('eta', a, b)
    n_max = max(n_order)
    m_max = max(m_order) - 2 * n_max

    old_coeff_grid = []
    for n in range(0, m_max + 2 * n_max + 1):
        old_coeff_grid.append([0] * (m_max + 2 * n_max + 1))
    old_coeff_grid[0][0] = 1
    order = 0

    for n in range(0, n_max + 1):
        for m in range(0, 2 * (n_max - n) + m_max + 1):
            # Hack implementation of the g(r(a, b), eta(a, b)) chain rule
            if n == 0 and m == 0:
                coeff_grid = deepcopy(old_coeff_grid)
            elif m == 0:
                for i in range(m + n - 1, -1, -1):
                    for j in range(m + n - i - 1, -1, -1):
                        coeff = old_coeff_grid[i][j]
                        if type(coeff) == type(1):
                            coeff_deriv = 0
                        else:
                            coeff_deriv = coeff.diff(b)
                        old_coeff_grid[i + 1][j] += coeff * r.diff(b)
                        old_coeff_grid[i][j + 1] += coeff * eta.diff(b)
                        old_coeff_grid[i][j] = coeff_deriv
                coeff_grid = deepcopy(old_coeff_grid)
            else:
                for i in range(m + n - 1, -1, -1):
                    for j in range(m + n - i - 1, -1, -1):
                        coeff = coeff_grid[i][j]
                        if type(coeff) == type(1):
                            coeff_deriv = 0
                        else:
                            coeff_deriv = coeff.diff(a)
                        coeff_grid[i + 1][j] += coeff * r.diff(a)
                        coeff_grid[i][j + 1] += coeff * eta.diff(a)
                        coeff_grid[i][j] = coeff_deriv

            # Replace r and eta derivatives with the rules found above
            deriv = deepcopy(coeff_grid)
            for l in range(order, 0, -1):
                for i in range(0, m + n + 1):
                    for j in range(0, m + n - i + 1):
                        if type(deriv[i][j]) != type(1):
                            deriv[i][j] = deriv[i][j].subs(Derivative(r, [a] * m_order[l] + [b] * n_order[l]), rules1[l])
                            deriv[i][j] = deriv[i][j].subs(Derivative(r, [b] * n_order[l] + [a] * m_order[l]), rules1[l])
                            deriv[i][j] = deriv[i][j].subs(Derivative(eta, [a] * m_order[l] + [b] * n_order[l]), rules2[l])
                            deriv[i][j] = deriv[i][j].subs(Derivative(eta, [b] * n_order[l] + [a] * m_order[l]), rules2[l])

            # Replace conformal block derivatives similarly for each spin
            for l in range(0, len(conformal_blocks)):
                new_deriv = 0
                for i in range(0, m + n + 1):
                    for j in range(0, m + n - i + 1):
                        new_deriv += deriv[i][j] * conformal_blocks[l].chunks[j].get(i, 0)
                table[l].vector.append(new_deriv.expand())
            order += 1
