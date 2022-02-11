import numpy as np
from io import StringIO
from pathlib import Path

# Note: Use **split** command to split a large text file into multiple
class TXTStream:
    def __init__(self, file, delimiter="\t"):
        super().__init__()
        input_path = Path(file)
        if input_path.is_dir():
            self.files = [str(p) for p in input_path.glob('*')]
        elif input_path.is_file():
            self.files = [file]

        self.delimiter = delimiter
        self.openfile = None
        self.openfile_id = -1
        self.open_next()

    def open_file(self, i):
        if self.openfile is not None and not self.openfile.closed:
            self.openfile.close()
        self.openfile = open(self.files[i], "r")
        self.openfileid = i
        
    def open_next(self):
        next_id = self.openfile_id + 1
        if next_id < len(self.files):
            self.open_file(next_id)
            return True
        else:
            return False

    def reset(self):
        self.open_file(0)

    def next_tik(self, batch=1):
        dims = 0
        X = []
        y = []
        end = False
        for _ in range(batch):
            line = self.openfile.readline()
            if line:
                row = np.genfromtxt(StringIO(line), delimiter=self.delimiter)
                if len(row) > 2:
                    if dims == 0:
                        dims = len(row) - 2
                    y.append(int(row[1]))
                    X.append(row[2:])
            else:
                self.openfile.close()
                end = not self.open_next()
                if end:
                    break

        return np.array(X), np.array(y), end


if __name__ == "__main__":
    stream = TXTStream("../data/tsv/agnews_manipulated.tsv")

    ended = False
    while not ended:
        X, y, ended = stream.next_tik(5)
        print(X)

