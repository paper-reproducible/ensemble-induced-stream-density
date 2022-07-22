from sys import argv
from Common import call_by_argv
from Scripts.benchmark_agnews import benchmark_agnews
from Scripts.benchmark_gaussian2x2 import benchmark_gaussian2x2
from Scripts.learning_curve import learning_curve
from ArtificialStream._plot import animate2d
from ArtificialStream._stream import ProbabilityStream
from ArtificialStream._gaussians import gaussian2x2

if __name__ == "__main__":
    if len(argv) < 2:
        benchmark_gaussian2x2()
    elif argv[1] == "agnews":
        benchmark_agnews()
    elif argv[1] == "animation":
        stream = ProbabilityStream()
        name = gaussian2x2(stream)
        _, _ = animate2d(stream, block=True, show_full=True, print_data=True)
    elif argv[1] == "learning_curve":
        call_by_argv(learning_curve, 2)
