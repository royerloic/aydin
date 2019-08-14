import os

#### THIS MUST BE CALLED BEFORE PLAIDML OR KERAS ARE IMPORTED!!!
from aydin.providers.opencl.opencl_provider import OpenCLProvider
from aydin.util.log.logging import lprint, lsection

os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"

import plaidml
from fuzzywuzzy import process


class PlaidMLProvider:
    def __init__(self, includes=[], excludes=['CPU']):

        with lsection(f"Initialising PlaidML device and context:"):
            self.context = plaidml.Context()
            plaidml.quiet()

            self.device = self.get_best_device(includes, excludes)

            lprint(f"Selected device: {self.device.description.decode()}")

            plaidml.settings.device_ids = [self.device.id.decode()]

    def get_all_devices(self):

        plaidml.settings._setup_for_test(plaidml.settings.user_settings)
        plaidml.settings.experimental = False
        devices, _ = plaidml.devices(self.context, limit=100, return_all=True)

        return devices

    def get_filtered_device_list(self, includes=[], excludes=[]):

        devices = self.get_all_devices()

        with lsection(f"All PlaidML devices:"):
            for device in devices:
                lprint(device.description.decode())

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

        with lsection(f"Filtered and sorted OpenCL devices:"):
            for device in devices:
                lprint(f"Device {device.description.decode()}")

        return list(devices)

    def get_best_device(self, includes=[], excludes=['CPU']):
        opencl = OpenCLProvider()
        filtered_devices = self.get_filtered_device_list(includes, excludes)

        best_device_name = opencl.device.name
        self.device_max_mem = opencl.device.global_mem_size

        filtered_device_names = [
            device.description.decode() for device in filtered_devices
        ]

        ratios = process.extract(best_device_name, filtered_device_names)
        best_match, score = process.extractOne(best_device_name, filtered_device_names)
        lprint(f"Best match: {best_match} with score: {score}")

        for device in filtered_devices:
            if device.description.decode() == best_match:
                return device

        return filtered_devices[0]

    def test_device(self, device):

        with lsection(f"Testing PlaidML device: {device.description.decode()} "):
            try:
                plaidml_device = plaidml.Device(self.context, device)

                matmul = plaidml.Function(
                    "function (B[X,Z], C[Z,Y]) -> (A) { A[x,y : X,Y] = +(B[x,z] * C[z,y]); }"
                )

                shape = plaidml.Shape(self.context, plaidml.DType.FLOAT32, 3, 3)
                a = plaidml.Tensor(plaidml_device, shape)
                b = plaidml.Tensor(plaidml_device, shape)
                c = plaidml.Tensor(plaidml_device, shape)
                plaidml.run(
                    self.context, matmul, inputs={"B": b, "C": c}, outputs={"A": a}
                )
                plaidml_device.close()

                lprint(f"Device {device.description.decode()} _is_ operational.")

                return True

            except Exception as e:

                lprint(e)
                lprint(
                    f"Device {device.description.decode()} is not operational: it failed to run some basic tensor operation."
                )

                return False
