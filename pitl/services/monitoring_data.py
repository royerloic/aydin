class MonitoringData:
    def __init__(self, monitoring_variables_emit):
        self.monitoring_variables_emit = monitoring_variables_emit

        self.monitoring_variables = None
        self._monitoring_image = None
        self._eval_metric = None
        self._iter = None

    @property
    def monitoring_variables(self):
        return self._iter, self._eval_metric, self._monitoring_image

    @monitoring_variables.setter
    def monitoring_variables(self, val):
        if val is not None:
            self._iter = val[0]
            self._eval_metric = val[1]
            self._monitoring_image = val[2]
            # print("changed...")
            # print(self.monitoring_variables)
            self.monitoring_variables_emit(
                (self._monitoring_image, self._eval_metric, self._iter)
            )
