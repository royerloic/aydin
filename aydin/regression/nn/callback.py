from keras.callbacks import Callback

from aydin.regression.regressor_base import RegressorCallbackTuple


class NNCallback(Callback):
    """
    """

    def __init__(self):
        super().__init__()

        self.regressor_callback = None
        self.iteration = 0

    def on_train_begin(self, logs=None):
        # print(f"on_train_begin...")
        pass

    def on_batch_ends(self, batch, logs=None):

        if self.regressor_callback:
            env = RegressorCallbackTuple()
            env.iteration = self.iteration
            env.model = self.model

            self.regressor_callback(env)
            self.iteration += 1

        pass

    def on_epoch_end(self, epoch, logs=None):
        # print(f"on_epoch_end...")
        current = self.get_monitor_value(logs)
        # print(f"VALUE = {current}")

    def on_train_end(self, logs=None):
        # print(f"on_train_end...")
        pass

    def get_monitor_value(self, logs):

        monitor_value = logs.get('val_loss')
        return monitor_value
