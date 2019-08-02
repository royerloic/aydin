from tqdm import tqdm


class ProgressBar(tqdm):
    def __init__(self, total=100):
        super(ProgressBar, self).__init__(total)

    def emit(self, val):
        self.update(val)
