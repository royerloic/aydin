from aydin.plaidml.plaidml_provider import PlaidMLProvider


def test_plaidml_provider():

    plaidml_provider = PlaidMLProvider()

    devices = plaidml_provider.get_all_devices()

    for dev in devices:
        print(f"Device:   {dev.id.decode()} : {dev.description.decode()} ")
