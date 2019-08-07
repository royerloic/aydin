from aydin.providers.opencl import OpenCLProvider


def demo_opencl_manager():
    provider = OpenCLProvider()

    assert provider.context is not None


demo_opencl_manager()
