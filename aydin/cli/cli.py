import atexit
import os
import sys
import click
import sentry_sdk

from aydin.gui.gui import run
from aydin.util.progress_bar import ProgressBar
from aydin.io.io import imwrite
from aydin.services.n2s import N2SService
from aydin.util.slicing_helper import apply_slicing
from aydin.util.resource import read_image_from_path
from aydin.util.update import get_latest_version_details, download_specific_version


CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])
VERSION = '0.0.5'


@click.group(invoke_without_command=True, context_settings=CONTEXT_SETTINGS)
@click.pass_context
@click.version_option(version=VERSION)
def aydin(ctx):
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # To disable TensorFlow logs
    sentry_sdk.init("https://d9d7db5f152546c490995a409023c60a@sentry.io/1498298")
    if ctx.invoked_subcommand is None:
        run(VERSION)
    else:
        pass


@aydin.command()
def update():
    print("Current version: ", VERSION)

    # Check updates and download if there is
    latest_version, latest_id = get_latest_version_details()

    if latest_version > VERSION:
        print(
            "Latest version: ",
            latest_version,
            "\nThere is a more recent version of Aydin, automatically updating and re-running now...",
        )
        # Download new version
        path_to_new_version = download_specific_version(latest_version, latest_id)

        # Run new version with same command and args
        args = click.get_os_args()
        words = [path_to_new_version] + args
        path_to_run = ' '.join(words)

        atexit.register(lambda: os.system(path_to_run))
        sys.exit()
    else:
        print("You are running the most updated version of Aydin")


@aydin.command()
@click.argument('path_source')
@click.option('-s', '--slicing', default='', type=str)
@click.option('-b', '--backend', default=None)
@click.option('--use-model/--save-model', default=False)
@click.option('--model-path', default=None)
def noise2self(**kwargs):

    # Check whether a path is provided for a model to use or save
    input_model_path = kwargs["model_path"] if kwargs["model_path"] else None

    # Get abspath to image and read it
    path = os.path.abspath(kwargs['path_source'])
    noisy, noisy_metadata = read_image_from_path(path)
    noisy = apply_slicing(noisy, kwargs['slicing'])

    # Run N2S service and save the result
    pbar = ProgressBar(total=100)
    n2s = N2SService(
        backend_preference=kwargs['backend'],
        use_model_flag=kwargs['use_model'],
        input_model_path=input_model_path,
    )
    denoised = n2s.run(noisy, pbar, noisy_metadata=noisy_metadata, image_path=path)
    path = path[:-4] + "_denoised" + path[-4:]
    with imwrite(path, shape=denoised.shape, dtype=denoised.dtype) as imarray:
        imarray[...] = denoised


# @aydin.command()
# @click.argument('train_source')
# @click.argument('train_truth')
# @click.argument('predict_target')
# @click.option('-s', '--slicing', default='', type=str)
# @click.option('-b', '--backend', default=None)
# @click.option('--save-model/--use-model', default=True)
# def noise2truth(**kwargs):
#     # Get abspath to images and read them
#     path_source = os.path.abspath(kwargs['train_source'])
#     path_truth = os.path.abspath(kwargs['train_truth'])
#     path_target = os.path.abspath(kwargs['predict_target'])
#
#     noisy = read_image_from_path(path_source)
#     noisy = apply_slicing(noisy, kwargs['slicing'])
#     truth = read_image_from_path(path_truth)
#     truth = apply_slicing(truth, kwargs['slicing'])
#     target = read_image_from_path(path_target)
#     target = apply_slicing(target, kwargs['slicing'])
#
#     # Run N2T service and save the result
#     pbar = ProgressBar(total=100)
#     n2t = N2TService(kwargs['backend'])
#     denoised = n2t.run(noisy, truth, target, pbar)
#     path = path_target[:-4] + "_denoised" + path_target[-4:]
#     with imwrite(path, shape=denoised.shape, dtype=denoised.dtype) as imarray:
#         imarray[...] = denoised


if __name__ == '__main__':
    aydin()
