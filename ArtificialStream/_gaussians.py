import numpy as np
from ._stream import ProbabilityStream


def ringnd(X, args):  # when radius=0 this is a gaussian
    # print(args)
    radius = args["radius"]
    bw = args["bandwidth"]
    o = args["center"]
    w = args["weight"]
    r = np.sum((X - o) ** 2, axis=1, keepdims=True) ** 0.5
    p = np.exp(-np.power(r - radius, 2.0) / (2 * np.power(bw, 2.0))) * w
    return np.squeeze(p)


def gaussian2x2(stream=None):
    if stream is None:
        stream = ProbabilityStream()
    else:
        stream.reset(reset_pdf=True)
    stream.append_pdf(
        ringnd,
        radius=0.0,
        bandwidth=0.1,
        center=np.array([0.25, 0.25]),
        weight=lambda t: 1 if t >= 0 else 0,
    )
    stream.append_pdf(
        ringnd,
        radius=0.0,
        bandwidth=0.1,
        center=np.array([0.75, 0.25]),
        weight=lambda t: 1 if t >= 1000 else 0,
    )
    stream.append_pdf(
        ringnd,
        radius=0.0,
        bandwidth=0.1,
        center=np.array([0.75, 0.75]),
        weight=lambda t: 1 if t >= 2000 else 0,
    )
    stream.append_pdf(
        ringnd,
        radius=0.0,
        bandwidth=0.1,
        center=np.array([0.25, 0.75]),
        weight=lambda t: 1 if t >= 3000 else 0,
    )
    return "gaussian2x2", [0, 1000, 2000, 3000], 4000


def gaussian4x4(stream=None):
    if stream is None:
        stream = ProbabilityStream()
    else:
        stream.reset(reset_pdf=True)
    stream.append_pdf(
        ringnd,
        radius=0.0,
        bandwidth=0.05,
        center=np.array([0.125, 0.125]),
        weight=lambda t: 1 if t >= 0 else 0,
    )
    stream.append_pdf(
        ringnd,
        radius=0.0,
        bandwidth=0.05,
        center=np.array([0.375, 0.125]),
        weight=lambda t: 1 if t >= 250 else 0,
    )
    stream.append_pdf(
        ringnd,
        radius=0.0,
        bandwidth=0.05,
        center=np.array([0.625, 0.125]),
        weight=lambda t: 1 if t >= 500 else 0,
    )
    stream.append_pdf(
        ringnd,
        radius=0.0,
        bandwidth=0.05,
        center=np.array([0.875, 0.125]),
        weight=lambda t: 1 if t >= 750 else 0,
    )
    stream.append_pdf(
        ringnd,
        radius=0.0,
        bandwidth=0.05,
        center=np.array([0.875, 0.375]),
        weight=lambda t: 1 if t >= 1000 else 0,
    )
    stream.append_pdf(
        ringnd,
        radius=0.0,
        bandwidth=0.05,
        center=np.array([0.625, 0.375]),
        weight=lambda t: 1 if t >= 1250 else 0,
    )
    stream.append_pdf(
        ringnd,
        radius=0.0,
        bandwidth=0.05,
        center=np.array([0.375, 0.375]),
        weight=lambda t: 1 if t >= 1500 else 0,
    )
    stream.append_pdf(
        ringnd,
        radius=0.0,
        bandwidth=0.05,
        center=np.array([0.125, 0.375]),
        weight=lambda t: 1 if t >= 1750 else 0,
    )
    stream.append_pdf(
        ringnd,
        radius=0.0,
        bandwidth=0.05,
        center=np.array([0.125, 0.625]),
        weight=lambda t: 1 if t >= 2000 else 0,
    )
    stream.append_pdf(
        ringnd,
        radius=0.0,
        bandwidth=0.05,
        center=np.array([0.375, 0.625]),
        weight=lambda t: 1 if t >= 2250 else 0,
    )
    stream.append_pdf(
        ringnd,
        radius=0.0,
        bandwidth=0.05,
        center=np.array([0.625, 0.625]),
        weight=lambda t: 1 if t >= 2500 else 0,
    )
    stream.append_pdf(
        ringnd,
        radius=0.0,
        bandwidth=0.05,
        center=np.array([0.875, 0.625]),
        weight=lambda t: 1 if t >= 2750 else 0,
    )
    stream.append_pdf(
        ringnd,
        radius=0.0,
        bandwidth=0.05,
        center=np.array([0.875, 0.875]),
        weight=lambda t: 1 if t >= 3000 else 0,
    )
    stream.append_pdf(
        ringnd,
        radius=0.0,
        bandwidth=0.05,
        center=np.array([0.625, 0.875]),
        weight=lambda t: 1 if t >= 3250 else 0,
    )
    stream.append_pdf(
        ringnd,
        radius=0.0,
        bandwidth=0.05,
        center=np.array([0.375, 0.875]),
        weight=lambda t: 1 if t >= 3500 else 0,
    )
    stream.append_pdf(
        ringnd,
        radius=0.0,
        bandwidth=0.05,
        center=np.array([0.125, 0.875]),
        weight=lambda t: 1 if t >= 3750 else 0,
    )
    return (
        "gaussian4x4",
        [
            0,
            250,
            500,
            750,
            1000,
            1250,
            1500,
            1750,
            2000,
            2250,
            2500,
            2750,
            3000,
            3250,
            3500,
            3750,
        ],
        4000,
    )


def gaussian4x4v2(stream=None):
    if stream is None:
        stream = ProbabilityStream()
    else:
        stream.reset(reset_pdf=True)
    stream.append_pdf(
        ringnd,
        radius=0.0,
        bandwidth=0.05,
        center=np.array([0.125, 0.125]),
        weight=lambda t: 1 if t >= 0 and t < 500 else 0,
    )
    stream.append_pdf(
        ringnd,
        radius=0.0,
        bandwidth=0.05,
        center=np.array([0.375, 0.125]),
        weight=lambda t: 1 if t >= 250 and t < 750 else 0,
    )
    stream.append_pdf(
        ringnd,
        radius=0.0,
        bandwidth=0.05,
        center=np.array([0.625, 0.125]),
        weight=lambda t: 1 if t >= 500 and t < 1000 else 0,
    )
    stream.append_pdf(
        ringnd,
        radius=0.0,
        bandwidth=0.05,
        center=np.array([0.875, 0.125]),
        weight=lambda t: 1 if t >= 750 and t < 1250 else 0,
    )
    stream.append_pdf(
        ringnd,
        radius=0.0,
        bandwidth=0.05,
        center=np.array([0.875, 0.375]),
        weight=lambda t: 1 if t >= 1000 and t < 1500 else 0,
    )
    stream.append_pdf(
        ringnd,
        radius=0.0,
        bandwidth=0.05,
        center=np.array([0.625, 0.375]),
        weight=lambda t: 1 if t >= 1250 and t < 1750 else 0,
    )
    stream.append_pdf(
        ringnd,
        radius=0.0,
        bandwidth=0.05,
        center=np.array([0.375, 0.375]),
        weight=lambda t: 1 if t >= 1500 and t < 2000 else 0,
    )
    stream.append_pdf(
        ringnd,
        radius=0.0,
        bandwidth=0.05,
        center=np.array([0.125, 0.375]),
        weight=lambda t: 1 if t >= 1750 and t < 2250 else 0,
    )
    stream.append_pdf(
        ringnd,
        radius=0.0,
        bandwidth=0.05,
        center=np.array([0.125, 0.625]),
        weight=lambda t: 1 if t >= 2000 and t < 2500 else 0,
    )
    stream.append_pdf(
        ringnd,
        radius=0.0,
        bandwidth=0.05,
        center=np.array([0.375, 0.625]),
        weight=lambda t: 1 if t >= 2250 and t < 2750 else 0,
    )
    stream.append_pdf(
        ringnd,
        radius=0.0,
        bandwidth=0.05,
        center=np.array([0.625, 0.625]),
        weight=lambda t: 1 if t >= 2500 and t < 3000 else 0,
    )
    stream.append_pdf(
        ringnd,
        radius=0.0,
        bandwidth=0.05,
        center=np.array([0.875, 0.625]),
        weight=lambda t: 1 if t >= 2750 and t < 3250 else 0,
    )
    stream.append_pdf(
        ringnd,
        radius=0.0,
        bandwidth=0.05,
        center=np.array([0.875, 0.875]),
        weight=lambda t: 1 if t >= 3000 and t < 3500 else 0,
    )
    stream.append_pdf(
        ringnd,
        radius=0.0,
        bandwidth=0.05,
        center=np.array([0.625, 0.875]),
        weight=lambda t: 1 if t >= 3250 and t < 3750 else 0,
    )
    stream.append_pdf(
        ringnd,
        radius=0.0,
        bandwidth=0.05,
        center=np.array([0.375, 0.875]),
        weight=lambda t: 1 if t >= 3500 else 0,
    )
    stream.append_pdf(
        ringnd,
        radius=0.0,
        bandwidth=0.05,
        center=np.array([0.125, 0.875]),
        weight=lambda t: 1 if t >= 3750 else 0,
    )
    return (
        "gaussian4x4v2",
        [
            0,
            250,
            500,
            750,
            1000,
            1250,
            1500,
            1750,
            2000,
            2250,
            2500,
            2750,
            3000,
            3250,
            3500,
            3750,
        ],
        4000,
    )


def gaussian2x2_ball(stream=None):
    if stream is None:
        stream = ProbabilityStream()
    else:
        stream.reset(reset_pdf=True)

    r = 1.0 / (1.0 + 2.0 ** 0.5)
    stream.append_pdf(
        ringnd,
        radius=0.0,
        bandwidth=0.1,
        center=np.array([-r, -r]),
        weight=lambda t: 1 if t >= 0 else 0,
    )
    stream.append_pdf(
        ringnd,
        radius=0.0,
        bandwidth=0.1,
        center=np.array([r, -r]),
        weight=lambda t: 1 if t >= 1000 else 0,
    )
    stream.append_pdf(
        ringnd,
        radius=0.0,
        bandwidth=0.1,
        center=np.array([r, r]),
        weight=lambda t: 1 if t >= 2000 else 0,
    )
    stream.append_pdf(
        ringnd,
        radius=0.0,
        bandwidth=0.1,
        center=np.array([-r, r]),
        weight=lambda t: 1 if t >= 3000 else 0,
    )
    return "gaussian2x2", [0, 1000, 2000, 3000], 4000
