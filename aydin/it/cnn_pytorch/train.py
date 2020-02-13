import copy
import math
import os

import torchvision.utils as vutils
from tensorboardX import SummaryWriter
from torch import nn
from torch.optim import SGD, Adam
from tqdm import tqdm

from mask import *
from util import psnr, clamp_tensor, ssim, mse


def callback_test(
    model, device, iteration, writer, test_batch, test_data_loader=None, masker=None
):
    with torch.no_grad():

        model.eval()

        result = {}

        if test_data_loader:
            mses = []
            psnrs = []
            ssims = []
            for X, Y in test_data_loader:
                X = X.to(device)
                Y = Y.to(device)

                if masker:
                    outputs = masker.infer_full_image(X, model)
                else:
                    outputs = model(X)

                mse_value = mse(outputs, Y, pad=2, rescale=True).mean().item()
                psnr_value = psnr(outputs, Y, pad=2, rescale=True).mean().item()
                ssim_value = ssim(outputs, Y).mean().item()

                mses.append(mse_value)
                if not (math.isnan(psnr_value) or math.isinf(psnr_value)):
                    psnrs.append(psnr_value)
                ssims.append(ssim_value)

            mean_mse = np.mean(mses)

            # PSNR is a useful transformation of mse, but averaging it perverse: if one image in the batch
            # is perfectly reconstructed, as can happen if we output all-zeros for a zero image, it biases the result.
            total_psnr = 10 * np.log10(1.0 / mean_mse)

            mean_psnr = np.mean(psnrs)
            mean_ssim = np.mean(ssims)

            result['mean_mse'] = mean_mse
            result['mean_psnr'] = mean_psnr
            result['mean_ssim'] = mean_ssim
            result['total_psnr'] = total_psnr

            writer.add_scalar('test mean mse', mean_mse, iteration)
            writer.add_scalar('test mean psnr', mean_psnr, iteration)
            writer.add_scalar('test mean ssim', mean_ssim, iteration)
            writer.add_scalar('test total psnr', total_psnr, iteration)

        X, Y = test_batch
        X = X.to(device)
        Y = Y.to(device)

        if masker:
            outputs = masker.infer_full_image(X, model)
        else:
            outputs = model(X)

        if not test_data_loader:
            writer.add_scalar(
                'test psnr',
                psnr(outputs, Y, pad=2, rescale=True).mean().item(),
                iteration,
            )

        y = vutils.make_grid(
            torch.cat([X, outputs, Y]),
            normalize=True,
            scale_each=False,
            range=(0, 1),
            nrow=len(X),
            padding=5,
        )

        writer.add_image('test_images', y, iteration)

        return result


def callback_validation(
    model,
    device,
    loss_function,
    iteration,
    writer,
    val_batch,
    val_data_loader=None,
    masker=None,
):
    """If we just have hold-out noisy data, for validation we
        1. Iterate over the validation dataset and compute average MSE.
        2. Compute and render full reconstructions for one validation batch.
    """

    with torch.no_grad():

        model.eval()

        result = {}

        if val_data_loader:
            losses = []
            if masker is None:
                for X, Y in val_data_loader:
                    X = X.to(device)
                    targets = Y.to(device)
                    outputs = model(X)
                    losses.append(loss_function(outputs, targets).item())
            else:
                for (X,) in val_data_loader:
                    X = X.to(device)
                    targets = X

                    if masker:
                        X, mask = masker.mask(X, 0, inference=True)
                        outputs = masker.infer_full_image(X, model)
                        losses.append(
                            loss_function(outputs * mask, targets * mask).item()
                        )

                    else:
                        outputs = masker.infer_full_image(X, model)
                        losses.append(loss_function(outputs, targets).item())

            val_loss = np.mean(losses)
            writer.add_scalar('val loss', val_loss, iteration)

            result['val_loss'] = val_loss

        if masker is None:
            X, Y = val_batch
            X = X.to(device)
            Y = Y.to(device)
            targets = Y
        else:
            X, = val_batch
            X = X.to(device)
            targets = X

        if masker:
            outputs = masker.infer_full_image(X, model)
        else:
            outputs = model(X)

        if not val_data_loader:
            val_loss = loss_function(outputs, targets)
            writer.add_scalar('val loss', val_loss.item(), iteration)

        y = vutils.make_grid(
            torch.cat([X, outputs, targets]),
            normalize=True,
            scale_each=False,
            range=(0, 1),
            nrow=len(X),
            padding=5,
        )

        writer.add_image('validation_images', y, iteration)

        return result


def train(
    train_data_loader,
    model,
    params,
    masker=None,
    val_data_loader=None,  # validation just has more noisy samples
    test_data_loader=None,  # test also has ground truth
):
    if params['loss'] == 'MSE':
        loss_function = nn.MSELoss()
    elif params['loss'] == "L1":
        loss_function = nn.L1Loss()
    elif params['loss'] == "SmoothL1":
        loss_function = nn.SmoothL1Loss()
    else:
        raise NotImplementedError

    if 'callback_frequency' in params:
        callback_frequency = params['callback_frequency']
    else:
        callback_frequency = 50

    if 'optimizer_params' not in params:
        params['optimizer_params'] = {}
    if params['optimizer'] == 'SGD':
        optimizer = SGD(model.parameters(), **params['optimizer_params'])
    elif params['optimizer'] == 'Adam':
        optimizer = Adam(model.parameters(), **params['optimizer_params'])
    else:
        raise NotImplementedError

    if 'tbfolder' in params:
        from datetime import datetime

        current_time = datetime.now().strftime('%b%d_%H-%M-%S')
        log_dir = os.path.join(
            params['tbfolder'], current_time + '_' + params.get('writer_suffix', '')
        )
        writer = SummaryWriter(log_dir=log_dir)
    else:
        writer = SummaryWriter(comment=params.get('writer_suffix', ''))

    device = params['device']

    model = model.to(device)

    if val_data_loader:
        val_batch = next(iter(val_data_loader))
    if test_data_loader:
        test_batch = next(iter(test_data_loader))

    test_results = {}
    best_results = {}
    best_model = model

    iteration = 0
    for epoch in range(params['num_epochs']):

        print("epoch %d" % epoch)

        if params['verbosity'] == 'low':
            loader = (
                train_data_loader
            )  # tqdm(train_data_loader, total=len(train_data_loader))
        else:
            loader = tqdm(train_data_loader, total=len(train_data_loader))

        for i, batch in enumerate(loader):

            model.train()

            if masker:
                (noisy_images,) = batch
            else:
                noisy_images, target_images = batch

            noisy_images = noisy_images.to(device)

            if masker:
                input_images, mask = masker.mask(noisy_images, iteration + i)

                output_images = model(input_images)

                # Mask input and output images:
                masked_outputs = output_images * mask
                masked_target_images = noisy_images * mask

                loss = loss_function(masked_outputs, masked_target_images)
                scalar_loss = loss.item() / mask.mean().item()

            else:
                target_images = target_images.to(device)
                output_images = model(noisy_images)
                loss = loss_function(output_images, target_images)
                scalar_loss = loss.item()

            # Clear gradients w.r.t. parameters
            optimizer.zero_grad()

            # Getting gradients w.r.t. parameters
            loss.backward()

            # Updating parameters
            optimizer.step()

            # Log scalars on Tensorboard:
            writer.add_scalar('loss', scalar_loss, iteration)

            # Log images on Tensorboard:
            iteration += 1

            # Log loss on screen

            print(iteration, " loss: ", scalar_loss)

            if iteration % callback_frequency == 0:

                if test_data_loader:
                    test_results = callback_test(
                        model,
                        device,
                        iteration,
                        writer,
                        test_batch,
                        test_data_loader=test_data_loader,
                        masker=masker,
                    )

                if val_data_loader:
                    val_results = callback_validation(
                        model,
                        device,
                        loss_function,
                        iteration,
                        writer,
                        val_batch,
                        val_data_loader=val_data_loader,
                        masker=masker,
                    )

                    if (
                        not best_results
                        or val_results['val_loss'] < best_results['val_loss']
                    ):
                        best_results = {**val_results, **test_results}
                        best_model = copy.deepcopy(model)
                        # if params['verbosity'] != 'low':
                        print(
                            "\n -----> New record: val_loss = %f "
                            % val_results['val_loss']
                        )

            if iteration > params['max_iter']:
                return best_model, best_results

    return best_model, best_results


def deep_image_prior(
    target,
    model,
    writer_name,
    device='cpu',
    truth=None,
    optimizer=None,
    learning_rate=0.01,
    loss_function=nn.MSELoss(),
    train_input=False,
    max_iter=100,
):
    """Takes in a target image, and trains the parameters of a model, with random input, to with that target.
    
    If train_input = True, then we also allow the input to change.
    
    Following https://github.com/DmitryUlyanov/deep-image-prior
    
    If there is a ground truth, we can track PSNR.
    
    If data has the standard form (X1, X2, Y), then this can be used on a single element via:
    
    dataset = 'MNIST'
    data, data_loader = load_dataset(dataset, lambda x: random_noise(x, 0.6, 'gaussian'))
    show_data(data[0])
    target = data[0][0].unsqueeze(0).to(device)
    truth = data[0][2].unsqueeze(0).to(device)

    deep_image_prior(target, Unet(1), 'test', truth, max_iter = 50)
    """

    model = model.to(device)
    target = target.to(device)

    input = torch.rand(target.shape).to(device)

    writer = SummaryWriter('runs/' + writer_name)

    if optimizer is None:
        if train_input:
            optimizer = torch.optim.SGD(
                list(model.parameters()) + list(input), lr=learning_rate, momentum=0.9
            )
        else:
            optimizer = torch.optim.SGD(
                model.parameters(), lr=learning_rate, momentum=0.9
            )

    if truth is None:
        truth = torch.zeros(target.shape).to(device)

    for iteration in range(max_iter):
        optimizer.zero_grad()

        output = model(input)
        loss = loss_function(output, target)
        loss.backward()

        optimizer.step()

        writer.add_scalar('loss', loss.item(), iteration)

        if iteration % 1 == 0:
            x = vutils.make_grid(
                torch.cat(
                    [clamp_tensor(output), clamp_tensor(target), clamp_tensor(truth)]
                ),
                normalize=False,
                scale_each=False,
                nrow=4,
            )

            writer.add_image('training_images', x, iteration)
            writer.add_scalar('psnr', psnr(output, truth).mean().item(), iteration)
