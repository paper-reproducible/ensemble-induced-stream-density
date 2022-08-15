import sys
import numpy as np
import pandas as pd
from pandasql import sqldf
from ._xp_utils import get_array_module


def set_printoptions():
    np.set_printoptions(formatter={"float_kind": "{:.4f}".format})


def func2obj(className, methodName, func, **kwargs):
    varList = func.__code__.co_varnames
    memberDict = {}

    def constructur(self):
        return

    memberDict["__init__"] = constructur
    for varName in varList:
        if varName in kwargs:
            memberDict[varName] = kwargs[varName]

    def exec(self, **kwargs):
        params = []
        for varName in varList:
            if varName in kwargs:
                params = params + [kwargs[varName]]
            elif varName in dir(self):
                params = params + [getattr(self, varName)]
        return func(*params)

    memberDict[methodName] = exec
    distClass = type(className, (object,), memberDict)
    return distClass()


def call_by_argv(func, start=1):
    args = []
    kwargs = {}
    parse_v = (
        lambda v: True
        if v == "True"
        else False
        if v == "False"
        else float(v)
        if v.isdecimal()
        else v
    )
    for arg_str in sys.argv[start:]:
        kv = arg_str.split("=")
        if len(kv) == 1:
            [v] = kv
            args.append(parse_v(v))

        if len(kv) == 2:
            [k, v] = kv
            kwargs[k] = parse_v(v)

    return func(*args, **kwargs)


def min_max_scale(X):
    xp, _ = get_array_module(X)
    X_ = X - xp.min(X, axis=0, keepdims=True)
    X_ = X_ / xp.max(X_, axis=0, keepdims=True)
    return X_


def query_pandas(sql, **kwargs):
    tables = {}
    for name in kwargs:
        value = kwargs[name]
        if isinstance(value, str) and value.endswith(".csv"):
            tables[name] = pd.read_csv(value)
        else:
            tables[name] = value  # assuming it is a data frame.
    results = sqldf(sql, tables)
    return results
