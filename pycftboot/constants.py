from symengine.lib.symengine_wrapper import RealMPFR, Symbol, zero, one, sqrt

cutoff = 0
prec = 1024
dec_prec = int((3.0 / 10.0) * prec)
tiny = RealMPFR("1e-" + str(dec_prec // 2), prec)

zero = zero.n(prec)
one = one.n(prec)
two = 2 * one
r_cross = 3 - 2 * sqrt(2).n(prec)

ell = Symbol('ell')
delta = Symbol('delta')
delta_ext = Symbol('delta_ext')
