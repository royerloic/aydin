import os
from collections import Counter
from pathlib import Path
from typing import List

import zarr

from aydin.util.log.log import lprint


def is_zarr_storage(input_path):
    try:
        z = zarr.open(input_path)
        assert len(z.shape) >= 0
        lprint(f"This path is a ZARR storage: {input_path}")
        # IF we reach this point, then we could open the file and therefore it is a Zarr file...
        return True
    except:
        lprint(f"This path is NOT a ZARR storage: {input_path}")
        return False


def get_files_with_most_frequent_extension(path) -> List[str]:

    files_in_folder = os.listdir(path)

    extensions = [Path(file).suffix[1:] for file in files_in_folder]

    counts = Counter(extensions)

    most_frequent_extension = sorted(counts, key=counts.__getitem__)[-1]

    files = [
        file for file in files_in_folder if file.endswith(f".{most_frequent_extension}")
    ]

    return files
