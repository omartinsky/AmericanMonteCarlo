import numpy


class TimeLine:
    def __init__(self, array):
        if isinstance(array, list):
            array = numpy.array(array)
        assert isinstance(array, numpy.ndarray)
        assert array[0] == 0, "Timeline must start with 0"
        self.__ndarray = array

    @property
    def ndarray(self):
        assert isinstance(self.__ndarray, numpy.ndarray)
        return self.__ndarray

    @property
    def length(self):
        return len(self.__ndarray)

    def __str__(self):
        return str(self.ndarray)