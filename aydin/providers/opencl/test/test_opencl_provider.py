from aydin.providers.opencl.opencl_provider import OpenCLProvider


def test_opencl_provider():

    opencl = OpenCLProvider()

    devices = opencl.get_all_devices()

    for device in devices:
        print(device)

    assert len(devices) > 0
