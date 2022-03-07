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
        if type(spin_irrep) == type(1):
            spin_irrep = [spin_irrep, 0]
        self.vector = derivatives
        self.label = spin_irrep
        self.poles = poles
