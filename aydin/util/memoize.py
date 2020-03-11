class Memoize:
    def __init__(self, function):
        self.fn = function
        self.memory = {}

    def __call__(self, *args):
        if args not in self.memory:
            self.memory[args] = self.fn(*args)
            return self.memory[args]
