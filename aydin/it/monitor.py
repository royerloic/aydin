class Monitor:
    def __init__(self, monitoring_callbacks=None, monitoring_images=None):
        if monitoring_callbacks is None:
            self.monitoring_callbacks = []
        else:
            self.monitoring_callbacks = monitoring_callbacks

        self.monitoring_images = monitoring_images

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
            # Update the values of variables to be monitored
            self._iter = val[0]
            self._val_loss = val[1]
            self._image = val[2]

            # Make calls to callbacks accordingly
            for monitoring_callback in self.monitoring_callbacks:
                monitoring_callback((self._iter, self._val_loss, self._image))
