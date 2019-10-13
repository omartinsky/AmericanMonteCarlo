# Copyright © 2018 Ondrej Martinsky, All rights reserved
# http://github.com/omartinsky/AmericanMonteCarlo

from copy import deepcopy
from typing import List

import numpy
from timeline import TimeLine


class PathGeneratorBase:
    def __init__(self, path_count: int, timeline: TimeLine):
        self.slices = list()
        self._path_count = path_count
        self.timeline = deepcopy(timeline)

    def get_slice(self, timeline_position):
        slice = self.slices[timeline_position]
        assert isinstance(slice, numpy.ndarray)
        return slice

    def get_slices(self):
        return self.slices

    @property
    def path_count(self):
        return self._path_count

    def calculate_quantiles(self, quantile_definitions: List):
        output = list()
        for slice in self.get_slices():
            sortedslice = sorted(slice)
            n = len(slice)
            output.append([sortedslice[min(n - 1, int(n * q))] for q in quantile_definitions])
        return output


class LogNormalPathGenerator(PathGeneratorBase):
    def __init__(self, timeline: TimeLine,
                 path_count: int,
                 s0: float,
                 drift: float,
                 sigma: float):
        super().__init__(path_count, timeline)
        s = numpy.ones(path_count) * s0
        self.slices.append(numpy.copy(s))
        for dt in numpy.diff(timeline.ndarray):
            dx = numpy.random.normal(0, 1, path_count) * dt ** .5
            ds = drift * s * dt + sigma * s * dx
            s += ds
            self.slices.append(numpy.copy(s))
