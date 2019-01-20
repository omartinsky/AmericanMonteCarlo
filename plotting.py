# Copyright © 2018 Ondrej Martinsky, All rights reserved
# http://github.com/omartinsky/AmericanMonteCarlo

from matplotlib import pyplot
import numpy
import itertools
from amc import *


def plot_paths(path_generator, n_paths_to_plot):
    assert isinstance(n_paths_to_plot, int)
    pyplot.gca().set_prop_cycle(None)
    slices_to_plot = numpy.matrix(path_generator.get_slices())[:, 0:n_paths_to_plot]
    pyplot.plot(path_generator.timeline.ndarray, slices_to_plot, linestyle='solid', alpha=0.2)


def plot_quantiles(path_generator, q):
    assert isinstance(q, list)
    quantiles = numpy.transpose(path_generator.calculate_quantiles(q))
    n = quantiles.shape[0]
    for i in range(int(n / 2)):  # iterate cols
        pyplot.fill_between(path_generator.timeline.ndarray,
                            quantiles[n - 1 - i],
                            quantiles[i],
                            alpha=(i + 1) / n / 2,
                            color='b',
                            label='%0.0f%% - %0.0f%% quantile' % (100 * q[i], 100 - 100 * q[i]))
    pyplot.legend()


def plot_cashflows(cash_flow_matrix, path_generator, **kwargs):
    pyplot.gca().set_prop_cycle(None)
    assert isinstance(cash_flow_matrix, CashFlowMatrix)
    count_paths = path_generator.path_count
    count_times = path_generator.timeline.length
    list_x = []
    list_y = []
    for iTime in range(count_times):
        t = path_generator.timeline.ndarray[iTime]
        cfmvalues, cfmtimes = cash_flow_matrix.find_next_cashflow(0)
        cash_flow_slice = numpy.where(cfmtimes == t, cfmvalues, 0)
        distribution_slice = path_generator.slices[iTime]
        for iPath in range(count_paths):
            payoff = cash_flow_slice[iPath]
            if payoff > 0:
                list_x.append(t)
                list_y.append(distribution_slice[iPath])
    pyplot.scatter(list_x, list_y, **kwargs)


def plot_regression(amc, t):
    x, y, itm_mask = amc.regression_data.data[t]
    interpolator = amc.interpolator_factory(x, y, itm_mask)
    x_filtered = list(itertools.compress(x, itm_mask))
    _linspace = numpy.linspace(min(x_filtered), max(x_filtered), 100)
    pyplot.scatter(x, y, linestyle='-', marker='.', color='grey', s=1)
    pyplot.plot(_linspace, interpolator.calc(_linspace), color='black', label='Regression at time t=%0.2f' % t)