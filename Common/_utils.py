import numpy as np


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
