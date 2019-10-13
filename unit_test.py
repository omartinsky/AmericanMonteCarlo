# Copyright © 2018 Ondrej Martinsky, All rights reserved
# http://github.com/omartinsky/AmericanMonteCarlo

import unittest

from teamcity import is_running_under_teamcity
from teamcity.unittestpy import TeamcityTestRunner

from amc import *
from logger import *
from pathgenerator import *


def create_cash_flow_matrix():
    cfm = CashFlowMatrix(5)
    cfm.add(3, numpy.array([1, 2, 3, 4, 5]))
    cfm.add(2, numpy.array([2, 1, 0, 0, 0]))
    cfm.add(1, numpy.array([0, 0, 0, 1, 1]))
    return cfm


class TestCashFlowMatrixNpv(unittest.TestCase):
    def test(self):
        cfm = create_cash_flow_matrix()
        rf = 0.02
        Z = lambda t: exp(-rf * t)
        actual = cfm.calc_npv(rf)
        expected = 2 * Z(2) + 1 * Z(2) + 3 * Z(3) + 1 * Z(1) + 1 * Z(1)
        expected /= 5
        self.assertEqual(expected, actual)


class TestCashFlowMatrixNextCashflow(unittest.TestCase):
    def test(self):
        cfm = create_cash_flow_matrix()
        P, T = cfm.find_next_cashflow(2)
        self.assertListEqual(list(P), [2, 1, 3, 0, 0])
        self.assertListEqual(list(T), [2, 2, 3, 0, 0])


class TestVectorOperations(unittest.TestCase):
    def test(self):
        numpy.warnings.filterwarnings('ignore')
        v1 = numpy.array([1, 2, 3, numpy.nan])
        v2 = numpy.array([numpy.nan, 2, 3, numpy.nan])
        print(v1 >= v2)


class PathGeneratorForTest(PathGeneratorBase):
    def __init__(self):
        super().__init__(8, TimeLine([0, 1, 2, 3]))
        self.slices = [
            numpy.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
            numpy.array([1.09, 1.16, 1.22, .93, 1.11, 0.76, .92, .88]),
            numpy.array([1.08, 1.26, 1.07, .97, 1.56, 0.77, 0.84, 1.22]),
            numpy.array([1.34, 1.54, 1.03, .92, 1.52, .90, 1.01, 1.34]),
        ]


class TestSimulator(unittest.TestCase):
    class Payoff(PayoffBase):
        def get(self, observable, t):
            return numpy.maximum(1.1 - observable, 0)

    def test(self):
        timeline = TimeLine([0., 1., 2., 3.])
        rf = 0.06
        logger = ConsoleLogger(LogLevel.ERROR)
        interpolator_factory = lambda x, y, itm_mask: PolynomialInterpolator.Create(x, y, itm_mask, 2)
        sim = AmcSimulator(TestSimulator.Payoff(), PathGeneratorForTest(), timeline, rf, interpolator_factory, logger)
        sim.run()
        values, times = sim.cfm.find_next_cashflow(0)
        expected_values = [0., 0., 0.07, 0.17, 0., 0.34, 0.18, 0.22]
        expected_times = [0., 0., 3., 1., 0., 1., 1., 1.]

        assert len(values) == len(times)
        n = len(values)
        for i in range(n):
            self.assertAlmostEqual(values[i], expected_values[i])
            self.assertAlmostEqual(times[i], expected_times[i])


class TestSimulator2(unittest.TestCase):
    class Payoff(PayoffBase):
        def get(self, observable, time):
            return numpy.maximum(1000 - observable, 0)

    def test(self):
        logger = ConsoleLogger(LogLevel.ERROR)
        timeline = TimeLine(numpy.linspace(0, 10, 100 + 1))
        rf = 0.02
        s0 = 1000
        random.seed(0)
        path_generator = LogNormalPathGenerator(timeline, path_count=10000, s0=s0, drift=rf, sigma=0.1)
        interpolator_factory = lambda x, y, itm_mask: PolynomialInterpolator.Create(x, y, itm_mask, 2)
        payoff = TestSimulator2.Payoff()
        sim = AmcSimulator(payoff, path_generator, timeline, rf, interpolator_factory, logger)
        sim.run()
        npv = sim.cash_flow_matrix.calc_npv(rf)
        logger.info("NPV: %f" % npv)
        self.assertEqual(npv, 64.897350224848651)


if __name__ == '__main__':
    if is_running_under_teamcity():
        runner = TeamcityTestRunner()
    else:
        runner = unittest.TextTestRunner()
    unittest.main(testRunner=runner)
