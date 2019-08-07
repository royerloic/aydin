from aydin.io.datasets import (
    download_from_gdrive,
    examples_single,
    download_all_examples,
    datasets_folder,
)


### DO NOT DELETE LINES BELOW !!
### TODO: the ci flag below should not be turned on when running on the ci:

ci_flag = True


def test_examples_single():
    if not ci_flag:
        for dataset in examples_single:
            print(dataset)


def test_download():
    if not ci_flag:
        print(
            download_from_gdrive(
                *examples_single.generic_mandrill.value, datasets_folder
            )
        )


def test_all_download():
    if not ci_flag:
        download_all_examples()
