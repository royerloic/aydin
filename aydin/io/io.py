import hashlib
import hashlib
import os
import traceback
from contextlib import contextmanager
from os import path
from os.path import join
from pathlib import Path

import dask
import numcodecs
import numpy
import pims
import skimage
import zarr
from czifile import czifile, CziFile
from tifffile import tifffile, TiffFile, memmap

## Axis codes:
#        'X' width, 'Y' height, 'S' sample, 'I' image series|page|plane,
#        'Z' depth, 'C' color|em-wavelength|channel, 'E' ex-wavelength|lambda,
#        'T' time, 'R' region|tile, 'A' angle, 'P' phase, 'H' lifetime,
#        'L' exposure, 'V' event, 'Q' unknown, '_' missing
import aydin
from aydin.io.folders import get_temp_folder
from aydin.io.utils import get_files_with_most_frequent_extension, is_zarr_storage
from aydin.util.log.logging import lsection, lprint


def is_batch(code, shape, axes):

    # special case:
    if len(shape) == 3 and 'X' in axes and 'Y' in axes and 'I' in code:
        return False

    return not code in 'XYZ'


class FileMetadata:
    is_folder = None
    extension = None
    axes = None
    shape = None
    dtype = None
    format = None
    batch_dim = None

    def __str__(self) -> str:
        return f" is_folder={self.is_folder}, ext={self.extension}, axes={self.axes}, shape={self.shape}, batch_dim={self.batch_dim}, dtype={self.dtype}, format={self.format} "


def analyse(input_path):

    with lsection("Analysing file at: {input_path}"):

        metadata = FileMetadata()

        metadata.is_folder = is_folder = os.path.isdir(input_path)
        metadata.extension = (Path(input_path).suffix)[1:]

        is_tiff = 'tif' in metadata.extension or 'tiff' in metadata.extension
        is_ome = is_tiff and 'ome.tif' in input_path
        is_czi = 'czi' in metadata.extension
        is_png = 'png' in metadata.extension
        is_zarr = 'zarr' in metadata.extension or is_zarr_storage(input_path)
        is_nd2 = 'nd2' in metadata.extension
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

        return metadata


def cache_as_zarr(input_path):

    with lsection("Caching file at: {input_path} as ZARR"):
        if is_zarr_storage(input_path):
            return input_path

        # We compute the hash from the beginning of the file (first 1M)
        with open(input_path, "rb") as f:
            # read the beginning of the file:
            bytes = f.read(1024 * 1024)
            hash = hashlib.sha256(bytes)
            readable_hash = hash.hexdigest()

        # We collect some information to create the cache file name:
        base_name = os.path.basename(input_path)
        base_name = base_name[0 : min(20, len(base_name))]
        last_modified = str(os.path.getmtime(input_path)).replace('.', '_')
        last_modified = last_modified[-min(20, len(last_modified) - 1) :]

        # Here is the file path:
        zarr_path = join(
            get_temp_folder(), f'{base_name}_{last_modified}_{readable_hash}.zarr'
        )
        lprint(f"Zarr file cache path is: {zarr_path}")

        if not path.exists(zarr_path):
            lprint(f"Zarr file cache does not exist we create it!")
            convert_to_zarr(input_path, zarr_path)
        else:
            lprint(f"Zarr file cache _does_ exist we use it...")

        return zarr_path


def convert_to_zarr(
    input_path,
    zarr_path,
    compression='zstd',
    compression_level=3,
    chunk_size=128,
    overwrite=False,
):
    with lsection("Converting file at: {input_path} to ZARR:"):

        lprint(f"destination zarr file: {zarr_path}")
        lprint(f"compression      ={compression}")
        lprint(f"compression_level={compression_level}")
        lprint(f"chunk_size       ={chunk_size}")
        lprint(f"overwrite        ={overwrite}")

        array, metadata = imread(input_path)

        mode = 'w' + ('' if overwrite else '-')
        filters = []  # [Delta(dtype='i4')]
        compressor = numcodecs.blosc.Blosc(
            cname=compression,
            clevel=compression_level,
            shuffle=numcodecs.blosc.Blosc.BITSHUFFLE,
        )
        chunks = tuple(1 if is_batch else chunk_size for is_batch in metadata.batch_dim)

        try:
            z = zarr.open_array(
                store=zarr_path,
                mode=mode,
                shape=metadata.shape,
                dtype=metadata.dtype,
                chunks=chunks,
                filters=filters,
                compressor=compressor,
            )

            # TODO: break it down per batch dimensions to provide feedback...
            lprint(f"Writting ZARR file: {zarr_path} now...")
            z[...] = array[...]
            lprint(f"Done!")

            z.attrs['axes'] = metadata.axes

            del z

        except Exception as e:
            print(
                f"Problem: can't create target file/directory, most likely the target dataset already exists: {zarr_path}"
            )
            return None


def imread(input_path, zarr_cache=False):

    with lsection(f"Reading image file at: {input_path} (zarr_cache={zarr_cache})"):

        if zarr_cache:
            zarr_path = cache_as_zarr(input_path)
            return imread(zarr_path, zarr_cache=False)

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

        return (array, metadata)


@contextmanager
def imwrite(output_path, shape, dtype):

    array = memmap(output_path, shape=shape, dtype=dtype)
    try:
        yield array
        array.flush()
    finally:
        del array
        lprint(
            f"Flushing and writting all bytes to TIFF file {output_path}  (shape={shape}, dtype={dtype})"
        )
