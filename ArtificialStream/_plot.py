import numpy as np
import matplotlib.pyplot as plt


def plot(l_x, l_y, l_label=None, show=True):
    fig = plt.figure()
    ax = plt.axes()
    ax.plot(l_x, l_y, linewidth=1, label=l_label)
    plt.grid(True)
    if show:
        plt.show()
    return fig, ax


def scatter(l_x, l_y, l_label=None, show=True, block=True, fig=None, ax=None):
    if fig is None:
        fig = plt.figure()
    if ax is None:
        ax = plt.axes()
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
    if l_label is not None:
        ax.scatter(l_x, l_y, s=0.5, c=l_label)
    else:
        ax.scatter(l_x, l_y, s=0.5, c="blue")
    if show:
        plt.show(block=block)
    return fig, ax


def animate2d(
    stream,
    batch_tiks=200,
    batch_round=25,
    pause=0.01,
    block=True,
    show_full=True,
    print_data=False,
):
    plt.ion()
    stream.reset()
    X, c, _ = stream.next_tik(print_data)
    if X is not None:
        x = X[:, 0]
        y = X[:, 1]
        fig, ax = scatter(x, y, c, show=False)
    else:
        fig = None
        ax = None

    for _ in range(batch_round):
        xj = None
        cj = None
        for _ in range(batch_tiks):
            xi, ci, _ = stream.next_tik(print_data)
            xj = xi if xj is None else np.append(xj, xi, axis=0)
            cj = np.array([ci]) if cj is None else np.append(cj, np.array([ci]), axis=0)
        if xj is not None:
            X = xj if X is None else np.append(X, xj, axis=0)
            x = xj[:, 0]
            y = xj[:, 1]
            if fig is not None and ax is not None:
                scatter(x, y, cj, show=False, fig=fig, ax=ax)
            else:
                fig, ax = scatter(x, y, cj, show=False)
            plt.draw()
            plt.pause(pause)
    if show_full:
        scatter(X[:, 0], X[:, 1], stream.current_truth(X), show=False)
    plt.show(block=block)
    return X, stream.count_retrieved
