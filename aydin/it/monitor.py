class Monitor:
    def __init__(self, monitoring_callback):
        self.monitoring_callback = monitoring_callback

        self.variables = None
        self._image = None
        self._val_loss = None
        self._iter = None

    @property
    def variables(self):
        return self._iter, self._val_loss, self._image

    @variables.setter
    def variables(self, val):
        if val is not None:
            self._iter = val[0]
            self._val_loss = val[1]
            self._image = val[2]
            # print("changed...")
            # print(self.monitoring_variables)
            self.monitoring_callback((self._iter, self._val_loss, self._image))
