from .common import coefficients, build_polynomial
from .constants import prec, tiny, delta


class PolynomialVector:
    """
    The main class for vectors on which the functionals being found by SDPB may act.

    Attributes
    ----------
    vector: A list of the components, expected to be polynomials in `delta`. The
            number of components is dictated by the number of derivatives kept in
            the search space.
    label:  A two element list where the first element is the spin and the second
            is a user-defined label for the representation of some global symmetry
            (or 0 if none have been set yet).
    poles:  A list of roots of the common denominator shared by all entries in
            `vector`. This allows one to go back to the original rational functions
            instead of the more convenient polynomials.
    """

    def __init__(self, derivatives, spin_irrep, poles):
        if isinstance(spin_irrep, int):
            spin_irrep = [spin_irrep, 0]

        self.vector = derivatives
        self.label = spin_irrep
        self.poles = poles

    def __eq__(self, other):
        if not isinstance(other, PolynomialVector):
            # don't attempt to compare against unrelated types
            return NotImplemented
        return self.__dict__ == other.__dict__

    def cancel_poles(self):
        """Checks which roots of a conformal block denominator are also roots of the
        numerator. Whenever one is found, a simple factoring is applied.
        """
        poles = []
        zero_poles = []
        for p in self.poles:
            if abs(float(p)) > tiny:
                poles.append(p)
            else:
                zero_poles.append(p)
        poles = zero_poles + poles

        for p in poles:
            # We should really make sure the pole is a root of all numerators
            # However, this is automatic if it is a root before differentiating
            if abs(self.vector[0].subs(delta, p)) < tiny:
                self.poles.remove(p)

                # A factoring algorithm which works if the zeros are first
                for n in range(0, len(self.vector)):
                    coeffs = coefficients(self.vector[n])
                    if abs(p) > tiny:
                        new_coeffs = [coeffs[0] / (-p).evalf(prec)]
                        for i in range(1, len(coeffs) - 1):
                            new_coeffs.append((new_coeffs[i - 1] - coeffs[i]) / p.evalf(prec))
                    else:
                        coeffs.remove(coeffs[0])
                        new_coeffs = coeffs

                    self.vector[n] = build_polynomial(new_coeffs)
