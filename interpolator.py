import numpy
from matplotlib import pyplot
import numpy.polynomial.chebyshev as chebyshev
import numpy.polynomial.polynomial as polynomial
import itertools


class ChebyshevInterpolator:
    def __init__(self, params):
        self.params = params

    def calc(self, xa):
        return chebyshev.chebval(xa, self.params)

    @staticmethod
    def Create(x, y, itm_mask, degree):
        # filter out-of-money paths
        x_filtered = list(itertools.compress(x, itm_mask))
        y_filtered = list(itertools.compress(y, itm_mask))

        interpolator = ChebyshevInterpolator(chebyshev.chebfit(x_filtered, y_filtered, degree))
        return interpolator


class PolynomialInterpolator:
    def __init__(self, params):
        self.params = params

    def calc(self, xa):
        return polynomial.polyval(xa, self.params)

    @staticmethod
    def Create(x, y, itm_mask, degree):
        # filter out-of-money paths
        x_filtered = list(itertools.compress(x, itm_mask))
        y_filtered = list(itertools.compress(y, itm_mask))

        interpolator = PolynomialInterpolator(polynomial.polyfit(x_filtered, y_filtered, degree))
        return interpolator
