from matplotlib.pyplot import axis
import numpy as np

from numbers import Number
from types import LambdaType, FunctionType
from Common import ball_samples


class ProbabilityStream:
    def __init__(
        self,
        n_features=2,
        init_count=0,
        increment=1,
        base_count=10000,
        ball=False,
        xp=np,
        linalg=np.linalg,
    ):
        super().__init__()

        self.n_features = n_features
        self.init_count = init_count
        self.increment = increment
        self.base_count = base_count
        self.ball = ball
        self.xp = xp
        self.linalg = linalg

        self.reset(reset_pdf=True)

    def reset(self, reset_pdf=False):
        self.t = 0
        self.count_retrieved = 0
        if reset_pdf is True:
            self.pdf = []
        return

    def append_pdf(self, pdf, **kwargs):
        self.pdf.append({"pdf": pdf, "kwargs": kwargs})

    def _calc_unnormalised_densities(self, X, t=0):
        results = []
        for i in range(len(self.pdf)):
            o = self.pdf[i]
            kwargs = {}
            for k in o["kwargs"]:
                arg = o["kwargs"][k]
                if isinstance(arg, (LambdaType, FunctionType)):
                    kwargs[k] = arg(t)
                else:
                    kwargs[k] = arg
            results.append(o["pdf"](X, kwargs))
        return results

    def _get_increment(self, t):
        if t == 0:
            return self.init_count
        elif isinstance(self.increment, Number):
            return self.increment
        elif isinstance(self.increment, LambdaType) or isinstance(
            self.increment, FunctionType
        ):
            return self.increment(t)
        else:
            raise ValueError(
                "Invalid increment value or function: {}".format(self.increment)
            )

    def _generate(self, t):
        xp = self.xp
        n = self._get_increment(t)
        if n == 0:
            return None, None, 0

        if self.ball:
            X = ball_samples(
                self.base_count, self.n_features, xp=self.xp, linalg=self.linalg
            )
        else:
            X = xp.random.rand(self.base_count, self.n_features)

        probs = xp.array(self._calc_unnormalised_densities(X, t))
        p = xp.sum(probs, axis=0)
        p = p / xp.sum(p)

        c = np.random.choice(self.base_count, n, p=p).tolist()
        X = xp.take(X, c, axis=0)
        probs = xp.take(probs, c, axis=1)
        y = xp.argmax(probs, axis=0)
        return X, y, n

    def current_truth(self, X):
        xp = self.xp
        probs = xp.array(self._calc_unnormalised_densities(X, self.t))
        y = xp.argmax(probs, axis=0)
        return y

    def next_tik(self, print_data=False):
        X, y, n = self._generate(self.t)
        if n > 0 and print_data:
            print("{}: {} => {}".format(self.t, X, y))
        self.t = self.t + 1
        self.count_retrieved = self.count_retrieved + n
        return X, y, n

    def calc_accumulated_density(self, X, t_to, t_from=0):
        xp = self.xp
        p = None
        for i in range(t_to - t_from):
            t = t_from + i
            n = self._get_increment(t)
            if n > 0:
                probst = self._calc_unnormalised_densities(X, t)
                pt = xp.sum(probst, axis=0)
                pt = pt / xp.sum(pt)
                p = pt * n if p is None else pt * n + p
        return p / xp.sum(p)
