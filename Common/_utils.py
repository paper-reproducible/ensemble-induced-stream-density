import numpy as np


def set_printoptions():
    np.set_printoptions(formatter={"float_kind": "{:.4f}".format})


def func2obj(className, methodName, sample_func, **kwargs):
    varList = sample_func.__code__.co_varnames
    memberDict = {}

    def constructur(self):
        return

    memberDict["__init__"] = constructur
    for varName in varList:
        memberDict[varName] = kwargs[varName]

    def sample(self):
        params = []
        for varName in varList:
            params = params + [getattr(self, varName)]
        return sample_func(*params)

    memberDict[methodName] = sample
    distClass = type(className, (object,), memberDict)
    return distClass()
