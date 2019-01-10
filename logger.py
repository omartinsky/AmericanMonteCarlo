# Copyright © 2018 Ondrej Martinsky, All rights reserved
# http://github.com/omartinsky/AmericanMonteCarlo

from enum import Enum


class LogLevel(Enum):
    ALL = 0
    DEBUG = 1
    INFO = 2
    WARNING = 3
    ERROR = 4

    def __ge__(self, other):
        return self.value >= other.value


class LoggerBase:
    def __init__(self, log_level):
        assert isinstance(log_level, LogLevel)
        self.log_level = log_level

    def debug(self, *args):
        self.print(*args, log_level=LogLevel.DEBUG)

    def info(self, *args):
        self.print(*args, log_level=LogLevel.INFO)

    def print(self, *args, log_level):
        raise NotImplementedError()

    def open(self):
        raise NotImplementedError()

    def close(self):
        raise NotImplementedError()


class FileLogger(LoggerBase):
    def __init__(self, filename, log_level):
        super().__init__(log_level)
        self.__filename = filename
        self.__file = None
        self.open()

    def open(self):
        if not self.__file:
            self.__file = open(self.__filename, 'w')

    def close(self):
        if self.__file:
            self.__file.close()
            self.__file = None

    def print(self, *args, log_level):
        if log_level >= self.log_level:
            print(*args, file=self.__file)
            self.__file.flush()
            # print(*args)

    def __enter__(self):
        self.open()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class ConsoleLogger(LoggerBase):
    def __init__(self, log_level):
        super().__init__(log_level)

    def print(self, *args, log_level):
        if log_level >= self.log_level:
            print(*args)

    def close(self):
        pass
