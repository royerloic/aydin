from keras.callbacks import Callback


class NNCallback(Callback):
    """
        Keras Callback to linkup to the it callback machinery
    """

    def __init__(self, regressor_callback):
        super().__init__()

        self.regressor_callback = regressor_callback
        self.iteration = 0

    def on_train_begin(self, logs=None):
        pass

    def on_batch_ends(self, batch, logs=None):
        self.notify(logs)

    def on_epoch_end(self, epoch, logs=None):
        self.notify(logs)

    def on_train_end(self, logs=None):
        self.notify(logs)

    def notify(self, logs):
        if self.regressor_callback:
            iteration = self.iteration
            val_loss = self.get_monitor_value(logs)
            model = self.model
            self.regressor_callback(iteration, val_loss, model)
            self.iteration += 1
        pass

    def get_monitor_value(self, logs):
        monitor_value = logs.get('val_loss')
        return monitor_value
