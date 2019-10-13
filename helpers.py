# Copyright © 2018 Ondrej Martinsky, All rights reserved
# http://github.com/omartinsky/AmericanMonteCarlo

import datetime
import json
from typing import List
import numpy

fmtstr_datetime = "%Y-%m-%dT%H:%M:%S.%fZ"
fmtstr_date = "%Y-%m-%d"
token_datetime = "$datetime"
token_date = "$date"


def generate_symmetric_quantiles(listing: List):
    return listing + [0.5] + [1 - q for q in reversed(listing)]


def json_serialise(obj, *args, **kwargs):
    return json.dumps(obj, default=json_serialise_handler, *args, **kwargs)


def json_serialise_handler(obj):
    if isinstance(obj, numpy.ndarray):
        return list(obj)
    if isinstance(obj, datetime.datetime):
        return {token_datetime: datetime.datetime.strftime(obj, fmtstr_datetime)}
    if isinstance(obj, datetime.date):
        return {token_date: datetime.date.strftime(obj, fmtstr_date)}
    raise TypeError(f"Unserialisable object of type {type(obj)}")
