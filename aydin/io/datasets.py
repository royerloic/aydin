import os
import zipfile
from enum import Enum
from os.path import join, exists

import gdown
import numpy
from skimage.exposure import rescale_intensity
from skimage.util import random_noise


from aydin.io import io
from aydin.io.folders import get_cache_folder
from aydin.util.log.log import lprint

datasets_folder = join(get_cache_folder(), 'data')

try:
    os.makedirs(datasets_folder)
except Exception:
    pass


def normalise(image):
    return rescale_intensity(
        image.astype(numpy.float32), in_range='image', out_range=(0, 1)
    )


def add_noise(image, intensity=5, variance=0.01, sap=0.0, dtype=numpy.float32):
    numpy.random.seed(0)
    noisy = numpy.random.poisson(image * intensity) / intensity
    noisy = random_noise(noisy, mode="gaussian", var=variance, seed=0)
    noisy = random_noise(noisy, mode="s&p", amount=sap, seed=0)
    noisy = noisy.astype(dtype)
    return noisy


# Convenience shortcuts:


def lizard():
    return examples_single.generic_lizard.get_array()


# def camera():
#    return examples_single.generic_camera.get_array()


def newyork():
    return examples_single.generic_newyork.get_array()


def pollen():
    return examples_single.generic_pollen.get_array()


def scafoldings():
    return examples_single.generic_scafoldings.get_array()


def characters():
    return examples_single.generic_characters.get_array()


def andromeda():
    return examples_single.generic_andromeda.get_array()


def fibsem(full=False):
    array = examples_single.scheffer_fibsem.get_array()
    if not full:
        array = array[0:1024, 0:1024]
    return array


class examples_single(Enum):
    def get_path(self):
        download_from_gdrive(*self.value, datasets_folder)
        return join(datasets_folder, self.value[1])

    def get_array(self):
        array, _ = io.imread(self.get_path())
        return array

    # XY natural images (2D monochrome):
    generic_crowd = ('13UHK8MjhBviv31mAW2isdG4G-aGaNJIj', 'crowd.tif')
    generic_mandrill = ('1B33ELiFuCV0OJ6IHh7Ix9lvImwI_QkR-', 'mandrill.tif')
    generic_newyork = ('15Nuu_NU3iNuoPRmpFbrGIY0VT0iCmuKu', 'newyork.png')
    generic_lizard = ('1GUc6jy5QH5DaiUskCrPrf64YBOLzT6j1', 'lizard.png')
    generic_pollen = ('1S0o2NWtD1shB5DfGRIqOFxTLOi8cHQD-', 'pollen.png')
    generic_scafoldings = ('1ZiWhHnkuaQH-BS8B71y00wkN1Ylo38nY', 'scafoldings.png')
    generic_andromeda = ('1Zl3DtkwUlZSbvpxGILexiIoLW1JOdJh8', 'andromeda.png')

    # Characters (2D monochrome, inverted):
    generic_characters = ('1ZWkHFI2iddKa9qv6tft4QZlCoDS5fLMK', 'characters.jpg')

    # XYC (RGB)
    celldiv = ('120w8j2XgJgwD0w0nqX-Gd0C4Qi_gJ8oO', 'Example-noisy1.png')

    # XY ()
    fmdd_hv115 = ('12C3_nW_wCFftKN0_XmGNoe-v72mp31p-', 'HV115_P0500510039.png')
    fmdd_hv110 = ('1B6WMgiaaUozgqwvHQtM0NwTUpmKuauKO', 'HV110_P0500510004.png')

    # XY
    scheffer_fibsem = (
        '1ZXRElGaq9Bshj-8sHKe95q-fH06bxQnM',
        'Scheffer_Dro_7column_iso04000.png',
    )

    # XY
    # metha_pol = (
    #     '13LqFk3elalzBbdo--8dIBQRBGTvrosDo',
    #     'Mehta_img_Polarization_t045_p000_z001.tif',
    # )
    metha_ret = (
        '1O6wfN404wSlKOoAwKck_pbjpX05qE7LH',
        'Mehta_img_Retardance_t045_p000_z001.tif',
    )
    metha_tra = (
        '1ZwOdD80EQspy5lKHEVw6s_ISf5H0kInF',
        'Mehta_img_Transmission_t045_p000_z001.tif',
    )

    leonetti_tm7sf2 = (
        '1HHsbZ6jyuJkIj6c7kGtsPKOgpUxo0ihw',
        'Leonetti_p4B3_1_TM7SF2_PyProcessed_IJClean.tif',
    )
    leonetti_sptssa = (
        '10kR7FSIyi7417XYTLrMJaGe3MfvMmSYA',
        'Leonetti_p1H10_2_SPTSSA_PyProcessed_IJClean.tif',
    )
    leonetti_snca = (
        '1UyF5HkZLwTaoiBf1sLHkTdw09yyCJyKO',
        'Leonetti_p1H8_2_SNCA_PyProcessed_IJClean.tif',
    )

    # XYZ
    keller_dmel = (
        '12DCAlDRSiTyGDSD7p06nk17GO3ztHg-Q',
        'SPC0_TM0132_CM0_CM1_CHN00_CHN01.fusedStack.tif',
    )

    janelia_flybrain = (
        '12Z6W_f3TqCsl_okKmaLcVUBgS6xEvdjj',
        'Flybrain_3ch_mediumSize.tif',
    )

    # 2D+t

    cognet_nanotube1 = (
        '1SmrBheUc6p5qTgtIEzedCwbN87HOW_O_',
        'Cognet_r03-s01-100mW-20ms-175 50xplpeg-173.tif',
    )

    cognet_nanotube_400fps = (
        '1T3M6MqHkSIqzAFqz2wzfYOXdtLnFxLp3',
        'Cognet_1-400fps.tif',
    )
    cognet_nanotube_200fps = (
        '1Z501FlQOBQmPaeBMCOGy6chBDh1bDjEf',
        'Cognet_1-200fps.tif',
    )
    cognet_nanotube_100fps = (
        '1T4UvbF3MRgT4jO4ExIHprvTqUXLiMjyA',
        'Cognet_1-100fps.tif',
    )

    # XYZT
    hyman_hela = ('12qOGxfBrnzrufgbizyTkhHipgRwjSIz-', 'Hyman_HeLa.tif')
    pourquie_elec = (
        '12VMZ6nphV9D40xiKYK6GpH9VghZM3IJC',
        '20190203_p4_electroporated.tif',
    )
    pourquie_quail = ('12SEwGxgFuCd9Oz6c8TbwOcXjOCXUx-aB', '20181228_p1_quail.tif')
    gardner_org = (
        '12MiulopOAa18o2haKZPfyA2piK7x9l3N',
        '405Col4Lm_488EpCAM_INS568_647GCGold_ImmSol_63x___-08.czi',
    )

    # XYZCT 1344 × 1024 × 1 × 1 × 93
    ome_mitocheck = ('1B9d8Yw_lidZg43U3VZAoalVHf9eHbCS7', '00001_01.ome.tiff')

    # XYZCT 160 × 220 × 8 × 2 × 12
    ome_spim = ('1BG6jCZGLEs1LDxKXjMqF0aV-iiqlushk', 'SPIM-ModuloAlongZ.ome.tiff')


class examples_zipped(Enum):
    def get_path(self):
        download_from_gdrive(
            *self.value,
            dest_folder=join(datasets_folder, os.path.splitext(self.value[1])[0]),
            unzip=True,
        )
        return join(datasets_folder, os.path.splitext(self.value[1])[0])

    care_tribolium = ('1BVNU-y9NJdNzkmsZcH8-2nhdhlRd4Mcw', 'tribolium.zip')
    unser_celegans = ('1D1I0LoA5LNsEr56kdJogik8uWN5wm2y3', 'celegans.zip')


def download_from_gdrive(
    id, name, dest_folder=datasets_folder, overwrite=False, unzip=False
):

    try:
        os.makedirs(dest_folder)
    except Exception:
        pass

    url = f'https://drive.google.com/uc?id={id}'
    output_path = join(dest_folder, name)
    if overwrite or not exists(output_path):
        lprint(f"Downloading file {output_path} as it does not exist yet.")
        gdown.download(url, output_path, quiet=False)

        if unzip:
            lprint(f"Unzipping file {output_path}...")
            zip_ref = zipfile.ZipFile(output_path, 'r')
            zip_ref.extractall(dest_folder)
            zip_ref.close()
            # os.remove(output_path)

        return output_path
    else:
        lprint(f"Not downloading file {output_path} as it already exists.")
        return None


def download_all_examples():
    for example in examples_single:
        print(download_from_gdrive(*example.value))

    for example in examples_zipped:
        download_from_gdrive(
            *example.value,
            dest_folder=join(datasets_folder, os.path.splitext(example.value[1])[0]),
            unzip=True,
        )


def downloaded_example(substring):
    for example in examples_single.get_list():
        if substring in example.value[1]:
            print(download_from_gdrive(*example.value))


def downloaded_zipped_example(substring):
    for example in examples_zipped:
        if substring in example.value[1]:
            download_from_gdrive(
                *example.value,
                dest_folder=join(
                    datasets_folder, os.path.splitext(example.value[1])[0]
                ),
                unzip=True,
            )
