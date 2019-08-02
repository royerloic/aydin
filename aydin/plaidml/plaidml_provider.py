import os

#### THIS MUST BE CALLED BEFORE PLAIDML OR KERAS ARE IMPORTED!!!
os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"

import plaidml
from fuzzywuzzy import process

from aydin.opencl.opencl_provider import OpenCLProvider


class PlaidMLProvider:
    def __init__(self, includes=[], excludes=['CPU']):

        self.context = plaidml.Context()
        plaidml.quiet()

        self.device = self.get_best_device(includes, excludes)

        print(f"Selected device: {self.device.description.decode()}")

        plaidml.settings.device_ids = [self.device.id.decode()]

    def get_all_devices(self):

        plaidml.settings._setup_for_test(plaidml.settings.user_settings)
        plaidml.settings.experimental = False
        devices, _ = plaidml.devices(self.context, limit=100, return_all=True)

        return devices

    def get_filtered_device_list(self, includes=[], excludes=[], sort_by_mem_size=True):

        devices = self.get_all_devices()
        # print(platforms)

        for exclude in excludes:
            devices = [
                device
                for device in devices
                if not exclude in device.description.decode()
            ]

        for include in includes:
            devices = [
                device for device in devices if include in device.description.decode()
            ]

        devices = [device for device in devices if self.test_device(device)]

        # print(devices)

        return list(devices)

    def get_best_device(self, includes=[], excludes=['CPU']):
        opencl = OpenCLProvider()
        filtered_devices = self.get_filtered_device_list(includes, excludes)

        best_device_name = opencl.device.name
        filtered_device_names = [
            device.description.decode() for device in filtered_devices
        ]

        ratios = process.extract(best_device_name, filtered_device_names)
        print(ratios)
        best_match, score = process.extractOne(best_device_name, filtered_device_names)
        print(f"Best match: {best_match} with score: {score}")

        for device in filtered_devices:
            if device.description.decode() == best_match:
                return device

        return None

    def test_device(self, device):

        try:
            device = plaidml.Device(self.context, device)

            matmul = plaidml.Function(
                "function (B[X,Z], C[Z,Y]) -> (A) { A[x,y : X,Y] = +(B[x,z] * C[z,y]); }"
            )

            shape = plaidml.Shape(self.context, plaidml.DType.FLOAT32, 3, 3)
            a = plaidml.Tensor(device, shape)
            b = plaidml.Tensor(device, shape)
            c = plaidml.Tensor(device, shape)
            plaidml.run(self.context, matmul, inputs={"B": b, "C": c}, outputs={"A": a})
            device.close()

            print(f"Device {device} _is_ operational.")

            return True

        except Exception as e:

            print(e)
            print(
                f"Device {device} is not operational: it failed to run some basic tensor operation."
            )

            return False
