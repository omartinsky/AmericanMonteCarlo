# Copyright © 2018 Ondrej Martinsky, All rights reserved
# http://github.com/omartinsky/AmericanMonteCarlo

import colorama
from abc import *
from logger import *
from numpy import *
from copy import deepcopy
from timeline import TimeLine
from cashflowmatrix import CashFlowMatrix
from interpolator import PolynomialInterpolator
from pathgenerator import *
import plotting
import timeit

from helpers import *

colorama.init(autoreset=True)


class PayoffBase:
    @abstractmethod
    def get(self, observable, time) -> numpy.ndarray:
        raise NotImplementedError()


class RegressionData:
    def __init__(self):
        self.data = dict()


class AmcSimulator:
    def __init__(self, payoff_function, path_generator, timeline, rf, interpolator_factory, logger,
                 store_regression_data=False):
        assert isinstance(payoff_function, PayoffBase)
        assert isinstance(path_generator, PathGeneratorBase)
        assert isinstance(timeline, TimeLine)
        assert isinstance(rf, float)
        assert isinstance(logger, LoggerBase)
        assert callable(interpolator_factory)
        self.interpolator_factory = interpolator_factory
        self.payoff_function = payoff_function
        self.path_generator = path_generator
        self.timeline = timeline
        self.rf = rf
        self.cfm = CashFlowMatrix(self.path_generator.path_count)
        self.logger = logger
        self.do_plotting = False
        self.regression_data = RegressionData() if store_regression_data else None

    @property
    def cash_flow_matrix(self):
        return self.cfm

    def run(self):
        numpy.warnings.filterwarnings('ignore')
        for it, t in reversed(list(enumerate(self.timeline.ndarray))):
            # it: index to the timeline
            # t:  timeline value
            self.logger.info("Time t=%f" % t)

            # S_t: Distribution slice for observable at t
            # P_t: Distribution slice for the payoff at t
            S_t = self.path_generator.get_slice(it)
            P_t = self.payoff_function.get(S_t, t)
            assert isinstance(P_t, numpy.ndarray)

            if it == self.timeline.length - 1:
                # We start by adding payoffs from all paths at the expiration date to the cash-flow matrix, assuming no early exercise.
                # Some of the cash-flows will be brought forward (early exercise) in later iterations of the algorithm.
                self.cfm.add(t, P_t)
                continue

            itm_mask = P_t > 0  # in-the-money mask (boolean) for all paths

            # Whether the option is in-the-money at time "t", the holder must decide whether to exercise immediately (e.g. at "t")
            # or continue until the final expiration time.

            t1 = self.timeline.ndarray[it + 1]

            # P_t1: Distribution slice for the discounted payoff
            P_t1, T = self.cash_flow_matrix.find_next_cashflow(t1)
            z = T - t

            # Regress payoff from t+td discounted to time t, against observable value at t
            y = P_t1 * exp(-self.rf * z)
            x = S_t

            if any(itm_mask):
                interpolator = self.interpolator_factory(x, y, itm_mask)
                value_if_continued = interpolator.calc(x)
                value_if_exercised = P_t
                exercise_mask = value_if_exercised > value_if_continued

                if self.regression_data:
                    self.regression_data.data[t] = (x, y, itm_mask)

                self.logger.debug('Continuation', value_if_continued)
                self.logger.debug('Exercise', value_if_exercised)
            else:
                # If all paths are out-of-money, there is nothing to be exercised
                exercise_mask = full(self.path_generator.path_count, False)

            self.logger.debug('IsExercised', exercise_mask)

            self.cfm.add(t, where(exercise_mask, P_t, 0))
