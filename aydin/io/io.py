import os
import traceback
from contextlib import contextmanager
from os.path import join
from pathlib import Path
import imageio
import dask
import numpy
import skimage
import zarr
from czifile import czifile, CziFile
from tifffile import tifffile, TiffFile, memmap

# Axis codes:
#        'X' width, 'Y' height, 'S' sample, 'I' image series|page|plane,
#        'Z' depth, 'C' color|em-wavelength|channel, 'E' ex-wavelength|lambda,
#        'T' time, 'R' region|tile, 'A' angle, 'P' phase, 'H' lifetime,
#        'L' exposure, 'V' event, 'Q' unknown, '_' missing

from aydin.io.utils import get_files_with_most_frequent_extension, is_zarr_storage
from aydin.util.log.log import lsection, lprint
import aydin


def is_batch(code, shape, axes):
    # special case:
    if len(shape) == 3 and 'X' in axes and 'Y' in axes and 'I' in code:
        return False

    return code not in 'XYZ'


def is_channel(code):
    return code == "C"


class FileMetadata:
    is_folder = None
    extension = None
    axes = None
    shape = None
    dtype = None
    format = None
    batch_dim = None
    channel_dim = None

    def __str__(self) -> str:
        return f" is_folder={self.is_folder}, ext={self.extension}, axes={self.axes}, shape={self.shape}, batch_dim={self.batch_dim}, channel_dim={self.channel_dim}, dtype={self.dtype}, format={self.format} "


def analyse(input_path):
    with lsection(f"Analysing file at: {input_path}"):

        metadata = FileMetadata()

        metadata.is_folder = is_folder = os.path.isdir(input_path)
        metadata.extension = ((Path(input_path).suffix)[1:]).lower()

        is_tiff = 'tif' in metadata.extension or 'tiff' in metadata.extension
        is_czi = 'czi' in metadata.extension
        is_png = 'png' in metadata.extension
        is_jpg = 'jpg' in metadata.extension or 'jpeg' in metadata.extension
        is_zarr = 'zarr' in metadata.extension or is_zarr_storage(input_path)
        is_npy = 'npy' in metadata.extension

        try:

            if is_zarr:
                lprint(f"Analysing file {input_path} as ZARR file")

                array = zarr.open(input_path)

                metadata.format = 'zarr'
                metadata.shape = array.shape
                metadata.dtype = array.dtype

                if 'axes' in array.attrs:
                    metadata.axes = array.attrs['axes']

            elif is_tiff:
                lprint(f"Analysing file {input_path} as TIFF file")

                metadata.format = 'tiff'

                with TiffFile(input_path) as tif:

                    if len(tif.series) >= 1:
                        serie = tif.series[0]
                        metadata.shape = serie.shape
                        metadata.dtype = serie.dtype
                        metadata.axes = serie.axes

                    else:
                        lprint(f'There is no series in file: {input_path}')

            elif is_czi:
                lprint(f"Analysing file {input_path} as CZI file")
                with CziFile(input_path) as czi:

                    metadata.format = 'czi'
                    metadata.shape = czi.shape
                    metadata.dtype = czi.dtype
                    metadata.axes = czi.axes

            elif is_png:
                lprint(f"Analysing file {input_path} as PNG file")
                array = skimage.io.imread(input_path)

                metadata.format = 'png'
                metadata.shape = array.shape
                metadata.dtype = array.dtype

                # TODO: Check order:
                metadata.axes = "YX"

            elif is_jpg:
                lprint(f"Analysing file {input_path} as JPEG file")
                array = skimage.io.imread(input_path)

                metadata.format = 'jpg'
                metadata.shape = array.shape
                metadata.dtype = array.dtype

                # TODO: Check order:
                metadata.axes = "YX"

            elif is_npy:
                lprint(f"Analysing file {input_path} as NPY file")

                array = numpy.load(input_path)
                metadata.format = 'czi'
                metadata.shape = array.shape
                metadata.dtype = array.dtype
                metadata.axes = ''.join(('Q',) * len(array.shape))

            elif is_folder:

                files = get_files_with_most_frequent_extension(input_path)
                files.sort()

                imread = dask.delayed(
                    aydin.io.io.imread, pure=True
                )  # Lazy version of imread

                lazy_images = [
                    imread(join(input_path, filename)) for filename in files
                ]  # Lazily evaluate imread on each path

                file_metadata = analyse(join(input_path, files[0]))

                arrays = [
                    dask.array.from_delayed(
                        lazy_image,  # Construct a small Dask array
                        dtype=file_metadata.dtype,  # for every lazy value
                        shape=file_metadata.shape,
                    )
                    for lazy_image in lazy_images
                ]

                array = dask.array.stack(arrays, axis=0)

                metadata.format = 'folder-' + file_metadata.format
                metadata.shape = array.shape
                metadata.dtype = array.dtype
                metadata.axes = 'Q' + file_metadata.axes

                metadata.array = array

                pass

        except Exception as error:
            print(error)
            print(traceback.format_exc())
            lprint(f"Could not analyse file {input_path} !")
            return None

        if metadata.axes:
            metadata.batch_dim = tuple(
                is_batch(axis, metadata.shape, metadata.axes) for axis in metadata.axes
            )

            metadata.channel_dim = tuple(is_channel(axis) for axis in metadata.axes)

        return metadata


def imread(input_path):
    with lsection(f"Reading image file at: {input_path}"):

        metadata = analyse(input_path)
        lprint(f"Metadata: {metadata}")

        try:
            try:
                if metadata.format == 'zarr':
                    lprint(f"Reading file {input_path} as ZARR file")
                    array = dask.array.from_zarr(input_path)
                elif metadata.format == 'tiff':
                    lprint(f"Reading file {input_path} as TIFF file")
                    array = tifffile.imread(input_path)
                elif metadata.format == 'czi':
                    lprint(f"Reading file {input_path} as CZI file")
                    array = czifile.imread(input_path)
                elif metadata.format == 'png':
                    lprint(f"Reading file {input_path} as PNG file")
                    array = skimage.io.imread(input_path)
                elif metadata.format == 'npy':
                    lprint(f"Reading file {input_path} as NPY file")
                    array = numpy.load(input_path)
                elif metadata.format.startswith('folder'):
                    lprint(f"Reading file {input_path} as Folder")
                    array = metadata.array
                else:
                    lprint(f"Reading file {input_path} using skimage imread")
                    array = skimage.io.imread(input_path)

            except Exception as error:
                lprint(error)
                lprint(traceback.format_exc())
                lprint(f"Reading file {input_path} using backup plan")
                array = skimage.io.imread(input_path)
        except Exception as error:
            lprint(error)
            lprint(traceback.format_exc())
            lprint(f"Could not read file {input_path} !")
            array = None

        # Remove single-dimensional entries from the array shape.
        # array = numpy.squeeze(array)

        return array, metadata


def imwrite(array, output_path, shape, dtype):
    if output_path[:-4] == ".tif" or output_path[:-4] == ".czi":
        with imwrite_contextmanager(output_path, shape, dtype) as save_array:
            output_path[:-4] = ".tif"
            save_array[...] = array
    else:
        imageio.imwrite(output_path, array)


@contextmanager
def imwrite_contextmanager(output_path, shape, dtype):
    array = memmap(output_path, shape=shape, dtype=dtype)
    try:
        yield array
        array.flush()
    finally:
        del array
        lprint(
            f"Flushing and writting all bytes to TIFF file {output_path}  (shape={shape}, dtype={dtype})"
        )
