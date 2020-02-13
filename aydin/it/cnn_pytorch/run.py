import yaml
from torch.utils.data import DataLoader

from datasets.datasets import *
from mask import Masker
from models.models import *
from train import train


def run(params, device):
    if 'train' in params['max_items']:
        train_data = get_dataset(
            mode="train",
            method=params['method'],
            max_items=params['max_items']['train'],
            **params['dataset'],
        )
    else:
        train_data = get_dataset(
            mode="train", method=params['method'], **params['dataset']
        )

    valid_data = get_dataset(
        mode="valid",
        method=params['method'],
        max_items=params['max_items']['valid'],
        **params['dataset'],
    )

    train_data_loader = DataLoader(
        train_data,
        batch_size=params['batch_size']['train'],
        shuffle=False,
        num_workers=3,
    )

    val_data_loader = DataLoader(
        valid_data,
        batch_size=params['batch_size']['valid'],
        shuffle=False,
        num_workers=3,
    )

    if params['dataset']['has_test']:
        test_data = get_dataset(
            mode="test",
            method=params['method'],
            max_items=params['max_items']['test'],
            **params['dataset'],
        )
        test_data_loader = DataLoader(
            test_data,
            batch_size=params['batch_size']['test'],
            shuffle=False,
            num_workers=3,
        )
    else:
        test_data_loader = None

    if 'writer_suffix' not in params['training']:
        params['training']['writer_suffix'] = '-'.join(
            [params['dataset']['name'], params['method']]
        )
    params['training']['device'] = device

    out_channels = train_data[0][0].shape[0]
    in_channels = out_channels + (1 if params['mask']['include_mask_as_input'] else 0)

    model = get_model(
        in_channels=in_channels, out_channels=out_channels, **params['model']
    ).to(device)

    if params['method'] == 'self':
        masker = Masker(3)
        if 'mask' in params:
            masker = Masker(**params['mask'])
            print(masker.random)
    else:
        masker = None

    return train(
        train_data_loader,
        model,
        params['training'],
        masker,
        val_data_loader=val_data_loader,
        test_data_loader=test_data_loader,
    )


def run_deep_image_prior():
    from train import deep_image_prior

    from datasets.disks import SyntheticDisks

    data = SyntheticDisks()
    for noise_params in [
        {'mode': 'gaussian_poisson', 'photons_at_max': 1.0, 'std': 0.2},
        {'mode': 'gaussian_poisson', 'photons_at_max': 10.0, 'std': 0.2},
        {'mode': 'gaussian_poisson', 'photons_at_max': 100.0, 'std': 0.2},
    ]:
        noisy = random_noise(data[0][2], noise_params)

        deep_image_prior(
            noisy.unsqueeze(0),
            Unet(1),
            'deep-image-prior-' + "-".join([str(v) for v in noise_params.values()]),
            truth=data[0][2].unsqueeze(0),
            max_iter=45,
        )


def run_deep_image_prior_immuno():
    from train import deep_image_prior
    from skimage import data as skdata
    from skimage.color import rgb2grey
    from torch import Tensor

    img = rgb2grey(skdata.immunohistochemistry())
    img = 1.0 - img

    x = Tensor(img).unsqueeze(0).unsqueeze(0)

    for noise_params in [
        {'mode': 'gaussian_poisson', 'photons_at_max': 1.0, 'std': 0.1},
        {'mode': 'gaussian_poisson', 'photons_at_max': 10.0, 'std': 0.1},
        {'mode': 'gaussian_poisson', 'photons_at_max': 100.0, 'std': 0.1},
    ]:
        noisy = random_noise(x, noise_params)

        deep_image_prior(
            noisy,
            Unet(1),
            'deep-image-prior-immuno-'
            + "-".join([str(v) for v in noise_params.values()]),
            truth=x,
            max_iter=45,
        )


# Train on GP, comparing to optimal PSNR and reconstruction.
#
# def run_gp():
#     device = "cuda:0"
#
#     size = 31
#     N = size * size
#     train_samples = 1000
#     noise_std = 0.5
#     length_scale = 2
#
#     train_data = gp.GPDataset(size, length_scale, train_samples, noise_std)
#     data_loader = DataLoader(train_data,
#                              batch_size=16,
#                              shuffle=True,
#                              num_workers=3)
#
#     val_samples = 32
#     val_data = gp.GPDataset(size, length_scale, val_samples, noise_std)
#     val_data_loader = DataLoader(train_data,
#                              batch_size=32,
#                              shuffle=False,
#                              num_workers=3)
#     val_batch = next(iter(val_data_loader))
#
#
#
#     loss_function = MSELoss()
#     learning_rate = 0.01
#     num_epochs = 1
#     shape = data[0][0].shape
#
#     for objective in ['truth', 'noise']:
#         writer_path = 'runs/' + '-'.join(['gp-', str(length_scale) +  + str(length_scale) '- '-' + objective])
#         if os.path.exists(writer_path):
#             print(writer_path)
#             shutil.rmtree(writer_path)
#
#         writer = SummaryWriter(writer_path)
#         model = Unet(shape[0]).to(device)
#         optimizer = SGD(model.parameters(), lr=learning_rate, momentum=0.9)
#
#         train(data_loader, val_batch, model, optimizer, loss_function,
#               writer, device, num_epochs, objective, 200, callback_truth, 10)
#


def template_variations():
    device = 'cuda:0'

    params['training']['num_epochs'] = 1
    params['method'] = 'truth'

    run(params, device)

    params['dataset'] = {
        "name": "synthetic-mnist",
        "noise_params": {'mode': 'bernoulli', 'p': 0.1},
        "has_test": True,
    }
    params['method'] = "noise"

    run(params, device)

    params['dataset'] = {"name": "mitofast", "has_test": False}
    params['method'] = "self"

    run(params, device)


if __name__ == "__main__":

    experiment = 'hanziL1'
    device = 'cuda:3'

    with open('experiments/' + experiment + '.yaml') as fp:
        params = yaml.load(fp.read())

    print(params['training'])

    for n in [64, 128, 256, 512, 1024, 2048, 4096]:
        for model_name in ['unet', 'baby-unet', 'convolution']:
            params['model']['name'] = model_name
            if model_name is 'convolution':
                params['model']['width'] = 11

            steps_per_epoch = max(1, n / 128)
            params['max_items']['train'] = n
            params['training']['num_epochs'] = int(500 // steps_per_epoch)
            params['training']['callback_frequency'] = min(20, 20 * 128 // n)
            params['training']['writer_suffix'] = (
                '-hanzi-sample-' + str(n) + '-model-' + model_name
            )

            print(params['training'])
            run(params, device)
