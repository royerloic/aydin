import atexit
import json
import os
import platform
import sys

import requests

import click
import logging

import sentry_sdk
from google_drive_downloader import GoogleDriveDownloader as gdd

from aydin.cli.progress_bar import ProgressBar
from aydin.gui import gui
from aydin.io.io import imwrite
from aydin.services.n2s import N2SService
from aydin.services.n2t import N2TService
from aydin.util.resource import read_image_from_path
from aydin.examples.demo_it_2D_cli import demo_aydin_2D


import plaidml.keras

plaidml.keras.install_backend()

logger = logging.getLogger(__name__)


def absPath(myPath):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    import sys

    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
        logger.debug(
            "found MEIPASS: %s " % os.path.join(base_path, os.path.basename(myPath))
        )

        return os.path.join(base_path, os.path.basename(myPath))
    except Exception as e:
        logger.debug("did not find MEIPASS: %s " % e)

        base_path = os.path.abspath(os.path.dirname(__file__))
        return os.path.join(base_path, myPath)


CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])
VERSION = '0.0.3'


@click.group(invoke_without_command=True, context_settings=CONTEXT_SETTINGS)
@click.pass_context
@click.version_option(version=VERSION)
def aydin(ctx):
    sentry_sdk.init("https://d9d7db5f152546c490995a409023c60a@sentry.io/1498298")
    if ctx.invoked_subcommand is None:
        # gui.run()
        print("Run aydin with a command please...")
    else:
        pass


@aydin.command()
def update(**kwargs):
    # Check if there is a new version
    print("this is a test version3 ", VERSION)
    r = requests.get("https://api.github.com/gists/b99bfed533738200a82dbf22d5406a9e")
    versions_json = json.loads(r.json()['files']['secret.json']['content'])
    latest_version = list(versions_json[str(platform.system())].keys())[-1]
    latest_id = versions_json[str(platform.system())][latest_version]

    if latest_version > VERSION:
        print(
            "There is a more recent version of Aydin, automatically updating and re-running now..."
        )
        atexit.register(update_app, "aydin_" + latest_version, latest_id)
        sys.exit()
    else:
        print("You are running the most updated version of Aydin")


def update_app(name, id):
    # Download newest version
    if platform.system() == 'Windows':
        name += ".exe"
    updated_app_path = os.path.join(str(os.getcwd()), name)
    gdd.download_file_from_google_drive(
        file_id=id, dest_path=updated_app_path, unzip=False
    )
    print("Please find the most recent version here: ", updated_app_path)

    if platform.system() == 'Darwin':
        os.system("chmod 755 " + updated_app_path)

    # Run same command with newest version
    args = click.get_os_args()
    words = [updated_app_path] + args
    os.system(' '.join(words))


@aydin.command()
@click.argument('mode')
def demo(**kwargs):
    if kwargs['mode'] == '2D':
        print("Running demo_aydin_2D")
        demo_aydin_2D()
    else:
        print("Rest of the demos not support by cli yet, sorry :(")


@aydin.command()
@click.argument('path_source')
def noise2self(**kwargs):
    # Get abspath to image and read it
    path = os.path.abspath(kwargs['path_source'])
    noisy = read_image_from_path(path)

    # Run N2S service and save the result
    pbar = ProgressBar(total=100)
    n2s = N2SService()
    denoised = n2s.run(noisy, pbar)
    path = path[:-4] + "_denoised" + path[-4:]
    with imwrite(path, shape=denoised.shape, dtype=denoised.dtype) as imarray:
        imarray[...] = denoised
    pbar.close()


@aydin.command()
@click.argument('train_source')
@click.argument('train_truth')
@click.argument('predict_target')
def noise2truth(**kwargs):
    # Get abspath to images and read them
    path_source = os.path.abspath(kwargs['train_source'])
    path_truth = os.path.abspath(kwargs['train_truth'])
    path_target = os.path.abspath(kwargs['predict_target'])
    noisy = read_image_from_path(path_source)
    truth = read_image_from_path(path_truth)
    target = read_image_from_path(path_target)

    # Run N2T service and save the result
    pbar = ProgressBar(total=100)
    n2t = N2TService()
    denoised = n2t.run(noisy, truth, target, pbar)
    path = path_target[:-4] + "_denoised" + path_target[-4:]
    with imwrite(path, shape=denoised.shape, dtype=denoised.dtype) as imarray:
        imarray[...] = denoised
    pbar.close()


if __name__ == '__main__':
    aydin()
