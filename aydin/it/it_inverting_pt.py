import os
from math import log1p, sqrt

import numpy
import scipy
import torch
from scipy.ndimage import median_filter
from torch import nn
from torch.utils.data import Dataset

from aydin.it.it_base import ImageTranslatorBase
from aydin.it.pytorch.models.RonnebergerUnet import RonnebergerUNet
from aydin.it.pytorch.models.basicunet import BasicUnet
from aydin.it.pytorch.models.feedforward import FeedForward
from aydin.it.pytorch.models.slimnet import SlimNet
from aydin.it.pytorch.models.unet import Unet
from aydin.util.log.log import lsection, lprint
from aydin.util.nd import extract_tiles


def to_numpy(tensor):
    return tensor.clone().detach().cpu().numpy()


class InvertingImageTranslator(ImageTranslatorBase):
    """
        Inverting image translator
    """

    def __init__(
        self,
        max_epochs=1024,
        patience=3,
        patience_epsilon=0.000001,
        learning_rate=0.01,
        batch_size=64,
        loss='l1',
        normaliser_type='percentile',
        balance_training_data=None,
        keep_ratio=1,
        max_voxels_for_training=4e6,
        monitor=None,
        use_cuda=True,
        device_index=0,
    ):
        """
        Constructs a CNN image translator using the pytorch deep learning library.

        :param normaliser_type: normaliser type
        :param balance_training_data: balance data ? (limits number training entries per target value histogram bin)
        :param monitor: monitor to track progress of training externally (used by UI)
        """
        super().__init__(normaliser_type, monitor=monitor)

        use_cuda = use_cuda and (torch.cuda.device_count() > 0)
        self.device = torch.device(f"cuda:{device_index}" if use_cuda else "cpu")
        lprint(f"Using device: {self.device}")

        self.max_epochs = max_epochs
        self.patience = patience
        self.patience_epsilon = patience_epsilon
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.loss = loss
        self.max_voxels_for_training = max_voxels_for_training
        self.keep_ratio = keep_ratio
        self.balance_training_data = balance_training_data

        self._stop_training_flag = False

        pytorch_info = torch.__config__.show().replace('\n', ';')
        lprint(f"PyTorch Info: {pytorch_info}")

    def save(self, path: str):
        """
        Saves a 'all-batteries-included' image translation model at a given path (folder).
        :param path: path to save to
        """
        with lsection(f"Saving 'classic' image translator to {path}"):
            frozen = super().save(path)
            pass
            # TODO: complete!

        return frozen

    def _load_internals(self, path: str):
        with lsection(f"Loading 'classic' image translator from {path}"):
            pass
            # TODO: complete!

    ## We exclude certain fields from saving:
    def __getstate__(self):
        state = self.__dict__.copy()
        del state['XXXXXXX']
        return state

    def get_receptive_field_radius(self, nb_dim):
        # TODO: estimate receptive field radius
        return 10

    def forward_model(self, images):
        pass
        # implement forward model: convolution with kernel

    def train(
        self,
        input_image,
        target_image,
        batch_dims=None,
        train_valid_ratio=0.1,
        callback_period=3,
    ):
        super().train(
            input_image,
            target_image,
            batch_dims=batch_dims,
            train_valid_ratio=train_valid_ratio,
            callback_period=callback_period,
        )

    def stop_training(self):
        lprint(f"Received notification to stop training loop now.")
        self._stop_training_flag = True

    def _train(
        self,
        input_image,
        target_image,
        batch_dims,
        train_valid_ratio=0.1,
        callback_period=3,
    ):
        self._stop_training_flag = False

        ndim = input_image.ndim

        # Dataset:
        tilesize = min(input_image.shape)
        mode = 'grid'
        dataset = self._get_dataset(
            input_image,
            target_image,
            self.self_supervised,
            tilesize=tilesize,
            mode=mode,
        )
        lprint(f"Tile generation mode: {mode}")
        lprint(f"Number tiles for training: {len(dataset)}")
        lprint(f"Tile dimensions: {tilesize}")

        # num_workers = max(3, os.cpu_count() // 2)
        num_workers = 0  # faster if data is already in memory...
        lprint(f"Number of workers for loading data: {num_workers}")
        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        )

        # Model
        self.model = SlimNet(num_channels=512, sf_factor=1, kernel_size=13, levels=5)
        self.model.set_trivial_basis()
        self.model.set_donut()
        # RonnebergerUNet(1, 1, depth=3, wf=4)

        self.model = self.model.to(self.device)
        number_of_parameters = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        lprint(f"Number of trainable parameters: {number_of_parameters}")

        # Optimiser:
        optimiser_class = torch.optim.Adam
        lprint(f"Optimiser: {optimiser_class}")
        lprint(f"Learning rate : {self.learning_rate}")
        optimizer = optimiser_class(
            self.model.trainable_parameters(),
            lr=self.learning_rate,
            weight_decay=0.1 * self.learning_rate,
        )

        # Loss functon:
        loss_function = nn.L1Loss()
        if self.loss.lower() == 'l2':
            lprint(f"Training loss: L2")
            loss_function = nn.MSELoss()
        elif self.loss.lower() == 'l1':
            loss_function = nn.L1Loss()
            lprint(f"Training loss: L1")

        # Masking function:
        density = 0.1
        lprint(f"Masking mode: random zero")

        def maskgen_function(shape, density):
            mask = (torch.FloatTensor(*shape).uniform_(0, 1) < density).float()
            return mask

        # Start training:
        self._train_loop(
            data_loader, optimizer, loss_function, maskgen_function, density=density
        )

    def _get_dataset(
        self, input_image, target_image, self_supervised, tilesize, mode='grid'
    ):
        class TrainingDataset(Dataset):
            def __init__(self, input_image, target_image, tilesize):
                """
                """

                self.input_tiles = extract_tiles(
                    input_image,
                    tile_size=tilesize,
                    extraction_step=tilesize,
                    flatten=True,
                )
                self.target_tiles = (
                    self.input_tiles
                    if self_supervised
                    else extract_tiles(
                        target_image,
                        tile_size=tilesize,
                        extraction_step=tilesize,
                        flatten=True,
                    )
                )

            def __len__(self):
                return len(self.input_tiles)

            def __getitem__(self, index):
                return (
                    self.input_tiles[index, numpy.newaxis, ...],
                    self.target_tiles[index, numpy.newaxis, ...],
                )

        if mode == 'grid':
            return TrainingDataset(input_image, target_image, tilesize)
        else:
            return None

    def _train_loop(
        self, data_loader, optimizer, loss_function, maskgen_function, density
    ):

        donut_filter = (1.0 / (3 * 3 - 1)) * numpy.array(
            [[1.0, 1.0, 1.0], [1.0, 0.0, 1.0], (1.0, 1.0, 1.0)]
        )
        donut_filter = donut_filter[numpy.newaxis, numpy.newaxis, :, :]
        donut_filter = torch.Tensor(donut_filter)
        donut_filter = donut_filter.to(self.device, non_blocking=True)

        with lsection(f"Training loop:"):
            lprint(f"Maximum number of epochs: {self.max_epochs}")
            lprint(
                f"Training type: {'self-supervised' if self.self_supervised else 'supervised'}"
            )

            for epoch in range(self.max_epochs):
                with lsection(f"Epoch {epoch}:"):

                    current_density = density / (1 + 0.1 * sqrt(epoch))
                    current_density = max(0.05, current_density)

                    loss_value = 0
                    iteration = 0

                    for i, (input_images, target_images) in enumerate(data_loader):
                        repeats = 1  # max(1, int(1.2/current_density))
                        lprint(
                            f"index: {i}, input shape:{input_images.shape}, target shape:{target_images.shape}, density:{current_density}, repeats:{repeats}"
                        )

                        # Clear gradients w.r.t. parameters
                        optimizer.zero_grad()

                        for repeat in range(0, repeats):
                            input_images_gpu = input_images.to(
                                self.device, non_blocking=True
                            )
                            target_images_gpu = target_images.to(
                                self.device, non_blocking=True
                            )
                            mask_gpu = maskgen_function(
                                input_images.shape, current_density
                            ).to(self.device, non_blocking=True)

                            masking = False

                            if masking:
                                inv_mask_gpu = 1.0 - mask_gpu
                                filtered_input_images_gpu = torch.nn.functional.conv2d(
                                    input_images_gpu, donut_filter, stride=1, padding=1
                                )
                                input_images_gpu = (
                                    input_images_gpu * inv_mask_gpu
                                    + filtered_input_images_gpu * (mask_gpu)
                                )

                            # Forward pass:
                            output_images_gpu = self.model(input_images_gpu)
                            # output_images = self.forward_model(output_images)

                            # loss:
                            if masking:
                                loss = loss_function(
                                    output_images_gpu * mask_gpu,
                                    target_images_gpu * mask_gpu,
                                )
                            else:
                                loss = loss_function(
                                    output_images_gpu, target_images_gpu
                                )

                            # Back-propagation:
                            loss.backward(retain_graph=repeat < repeats - 1)

                            # update training loss for whole image:
                            loss_value += loss.item()
                            iteration += 1

                            # Updating parameters
                            optimizer.step()

                        lprint(f"Training loss value: {loss_value/iteration}")

                        # import napari
                        # with napari.gui_qt():
                        #     viewer = napari.Viewer()
                        #     viewer.add_image(to_numpy(mask_gpu), name='mask_gpu')
                        #     viewer.add_image(to_numpy(input_images_gpu), name='input_images_gpu')
                        #     viewer.add_image(to_numpy(filtered_input_images_gpu), name='filtered_input_images_gpu')
                        #     viewer.add_image(to_numpy(masked_input_images_gpu), name='masked_input_images_gpu')
                        #     viewer.add_image(to_numpy(output_images_gpu), name='output_images_gpu')
                        #     viewer.add_image(to_numpy(target_images_gpu), name='target_images_gpu')

                    if self._stop_training_flag:
                        return

    def _translate(self, input_image, batch_dims=None):
        """
            Internal method that translates an input image on the basis of the trained model.
        :param input_image: input image
        :param batch_dims: batch dimensions
        :return:
        """

        input_image = torch.Tensor(input_image[numpy.newaxis, numpy.newaxis, ...])
        input_image = input_image.to(self.device)
        inferred_image: torch.Tensor = self.model(input_image)
        inferred_image = inferred_image.detach().cpu().numpy()
        inferred_image = inferred_image.squeeze()
        return inferred_image

    def _get_single_pixel_mask(self, shape, size, phasex, phasey):
        M = numpy.zeros(shape[-2:])
        for i in range(shape[-2]):
            for j in range(shape[-1]):
                M[i, j] = 1 if (i % size == phasex and j % size == phasey) else 0
        return torch.Tensor(M)

        # if epoch > 10 and j == 0:
        #     import napari
        #     with torch.no_grad():
        #         with napari.gui_qt():
        #             viewer = napari.Viewer()
        #             viewer.add_image(
        #                 to_numpy(mask), name='mask'
        #             )
        #             viewer.add_image(
        #                 to_numpy(1.0-mask), name='inv_mask'
        #             )
        #             viewer.add_image(
        #                 to_numpy(input_images), name='input_images'
        #             )
        #             viewer.add_image(
        #                 to_numpy(output_images), name='output_images'
        #             )
        #             viewer.add_image(
        #                 to_numpy(target_images), name='target_images'
        #             )
