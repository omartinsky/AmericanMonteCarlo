from typing import Union

import numpy


class CashFlowMatrix:
    def __init__(self, npaths: int):
        self.payoff_values = numpy.zeros(npaths)
        self.payoff_times = numpy.zeros(npaths)
        self.npaths = npaths

    def add(self,
            payoff_time: Union[int, float],
            payoff_values):
        assert isinstance(payoff_values, numpy.ndarray)
        payoff_mask = payoff_values > 0
        # use numpy.where to update only values where payoff_mask is set
        self.payoff_values = numpy.where(payoff_mask, payoff_values, self.payoff_values)
        self.payoff_times = numpy.where(payoff_mask, payoff_time, self.payoff_times)

    def find_next_cashflow(self, time: float):
        mask = self.payoff_times >= time
        values = numpy.where(mask, self.payoff_values, 0)
        times = numpy.where(mask, self.payoff_times, 0)
        return values, times

    def calc_npv(self, rf):
        values, times = self.find_next_cashflow(0)
        return numpy.average(values * numpy.exp(-rf * times))
