import numpy
from matplotlib import pyplot
import itertools


class PolynomialInterpolator:
    def __init__(self, params):
        assert type(params) == numpy.ndarray
        self.params = params
        self.n = len(self.params)
        self.range = numpy.array([i for i in reversed(range(self.n))])

    def _calc_number(self, x):
        return numpy.sum(x ** self.range * self.params)

    def calc(self, xa):
        vectorised = numpy.vectorize(lambda x : self._calc_number(x))
        return vectorised(xa)

    @staticmethod
    def CreateInterpolator(x, y, itm_mask, debug_plotting=False):
        assert isinstance(debug_plotting, bool)

        # filter out-of-money paths
        x_filtered = list(itertools.compress(x, itm_mask))
        y_filtered = list(itertools.compress(y, itm_mask))

        interpolator = PolynomialInterpolator(numpy.polyfit(x_filtered, y_filtered, 2))

        if debug_plotting:
            _linspace = numpy.linspace(min(x_filtered), max(x_filtered), 100)
            pyplot.scatter(x, y, linestyle='-', marker='.', color='grey', s=1)
            pyplot.plot(_linspace, interpolator.calc(_linspace), color='black'), pyplot.show()

        return interpolator
