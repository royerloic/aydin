from aydin.providers.opencl import OpenCLProvider


def test_opencl_provider():

    opencl = OpenCLProvider()

    assert len(opencl.get_all_devices()) != 0

    for device in opencl.get_all_devices():
        print(device)
