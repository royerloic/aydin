import math
from itertools import chain

import numpy
import torch
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset

from aydin.it.it_base import ImageTranslatorBase
from aydin.it.pytorch.models.feedforward import FeedForward
from aydin.it.pytorch.models.lucyrichardson import LucyRichardson
from aydin.it.pytorch.models.masking import Masking
from aydin.it.pytorch.models.psf_convolution import PSFConvolutionLayer
from aydin.it.pytorch.models.skipnet import SkipNet2D
from aydin.util.log.log import lsection, lprint
from aydin.util.nd import extract_tiles


def to_numpy(tensor):
    return tensor.clone().detach().cpu().numpy()


class DeconvolvingImageTranslator(ImageTranslatorBase):
    """
        Inverting image translator
    """

    def __init__(
        self,
        max_epochs=1024,
        patience=128,
        patience_epsilon=0.0,
        learning_rate=0.01,
        batch_size=64,
        loss='l1',
        normaliser_type='percentile',
        balance_training_data=None,
        keep_ratio=1,
        max_voxels_for_training=4e6,
        psf_kernel=None,
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
        self.psf_kernel = psf_kernel

        self.l1_weight_regularisation = 1e-9
        self.l2_weight_regularisation = 1e-9
        self.input_noise = 0.001
        self.reload_best_model_period = max_epochs
        self.reduce_lr_patience = 128
        self.reduce_lr_factor = 0.5
        self.denoising_model_class = SkipNet2D
        self.deconvolution_model_class = FeedForward  # LucyRichardson
        self.optimiser_class = Adam
        self.max_tile_size = 1024

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

    def train(
        self,
        input_image,
        target_image=None,
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

        shape = input_image.shape
        ndim = input_image.ndim

        # tile size:
        tile_size = min(self.max_tile_size, min(shape))

        # Decide on how many voxels to be used for validation:
        num_val_voxels = int(train_valid_ratio * input_image.size)
        lprint(
            f"Number of voxels used for validation: {num_val_voxels} (train_valid_ratio={train_valid_ratio})"
        )

        # Generate random coordinates for these voxels:
        val_voxels = tuple(numpy.random.randint(d, size=num_val_voxels) for d in shape)
        lprint(f"Validation voxel coordinates: {val_voxels}")

        # Training Tile size:
        train_tile_size = tile_size
        lprint(f"Train Tile dimensions: {train_tile_size}")

        # Prepare Training Dataset:
        train_dataset = self._get_dataset(
            input_image,
            target_image,
            self.self_supervised,
            tilesize=train_tile_size,
            mode='grid',
            is_training_data=True,
            validation_voxels=val_voxels,
        )
        lprint(f"Number tiles for training: {len(train_dataset)}")

        # Validation Tile size:
        val_tile_size = tile_size
        lprint(f"Validation Tile dimensions: {val_tile_size}")

        # Prepare Validation dataset:
        val_dataset = self._get_dataset(
            input_image,
            target_image,
            self.self_supervised,
            tilesize=val_tile_size,
            mode='grid',
            is_training_data=False,
            validation_voxels=val_voxels,
        )
        lprint(f"Number tiles for training: {len(val_dataset)}")

        # Training Data Loader:
        # num_workers = max(3, os.cpu_count() // 2)
        num_workers = 0  # faster if data is already in memory...
        lprint(f"Number of workers for loading training/validation data: {num_workers}")
        train_data_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=1,  # self.batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        )

        val_data_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=1,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        )

        # Denoising Model
        self.denoise_model = self.denoising_model_class(1, 1).to(self.device)
        if self.deconvolution_model_class is LucyRichardson:
            self.deconv_model = self.deconvolution_model_class(
                self.psf_kernel, 1, 1
            ).to(self.device)
        else:
            self.deconv_model = self.deconvolution_model_class(1, 1).to(self.device)
        # lprint(f"Model: {self.model}")

        number_of_parameters_denoise = sum(
            p.numel() for p in self.denoise_model.parameters() if p.requires_grad
        )
        lprint(
            f"Number of trainable parameters in denoise model: {number_of_parameters_denoise}"
        )
        number_of_parameters_deconv = sum(
            p.numel() for p in self.deconv_model.parameters() if p.requires_grad
        )
        lprint(
            f"Number of trainable parameters in deconv model: {number_of_parameters_deconv}"
        )
        lprint(
            f"Total number of trainable parameters in model: {number_of_parameters_deconv+number_of_parameters_denoise}"
        )

        self.psfconv = PSFConvolutionLayer(self.psf_kernel).to(self.device)
        self.masked_denoise_model = Masking(self.denoise_model).to(self.device)

        lprint(f"Optimiser class: {self.optimiser_class}")
        lprint(f"Learning rate : {self.learning_rate}")

        # Denoising optimiser:
        denoising_optimizer = self.optimiser_class(
            self.denoise_model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.l2_weight_regularisation,
        )
        lprint(f"Optimiser: {denoising_optimizer}")

        deconv_optimizer = self.optimiser_class(
            self.deconv_model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.l2_weight_regularisation,
        )
        lprint(f"Optimiser: {deconv_optimizer}")

        # Loss functon:
        loss_function = nn.L1Loss()
        if self.loss.lower() == 'l2':
            lprint(f"Training/Validation loss: L2")
            loss_function = nn.MSELoss(reduction='none')
        elif self.loss.lower() == 'l1':
            loss_function = nn.L1Loss(reduction='none')
            lprint(f"Training/Validation loss: L1")

        # Start training:
        self._train_loop(
            train_data_loader,
            val_data_loader,
            denoising_optimizer,
            deconv_optimizer,
            loss_function,
        )

    def _get_dataset(
        self,
        input_image: numpy.ndarray,
        target_image: numpy.ndarray,
        self_supervised: bool,
        tilesize: int,
        mode: str,
        is_training_data: bool,
        validation_voxels,
    ):
        class _Dataset(Dataset):
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

                mask_image = numpy.zeros_like(input_image)
                mask_image[validation_voxels[0], validation_voxels[1]] = 1

                self.mask_tiles = extract_tiles(
                    mask_image,
                    tile_size=tilesize,
                    extraction_step=tilesize,
                    flatten=True,
                )

            def __len__(self):
                return len(self.input_tiles)

            def __getitem__(self, index):
                input = self.input_tiles[index, numpy.newaxis, ...]
                target = self.target_tiles[index, numpy.newaxis, ...]
                mask = self.mask_tiles[index, numpy.newaxis, ...]

                return (input, target, mask)

        if mode == 'grid':
            return _Dataset(input_image, target_image, tilesize)
        else:
            return None

    def _train_loop(
        self,
        train_data_loader,
        val_data_loader,
        denoising_optimizer,
        deconv_optimizer,
        loss_function,
    ):

        # Scheduler:
        denoising_scheduler = ReduceLROnPlateau(
            denoising_optimizer,
            'min',
            factor=self.reduce_lr_factor,
            verbose=True,
            patience=self.reduce_lr_patience,
        )
        deconv_scheduler = ReduceLROnPlateau(
            deconv_optimizer,
            'min',
            factor=self.reduce_lr_factor,
            verbose=True,
            patience=self.reduce_lr_patience,
        )

        best_loss = math.inf
        best_denoise_model_state_dict, best_deconv_model_state_dict = None, None
        patience_counter = 0

        with lsection(f"Training loop:"):
            lprint(f"Maximum number of epochs: {self.max_epochs}")
            lprint(
                f"Training type: {'self-supervised' if self.self_supervised else 'supervised'}"
            )

            for epoch in range(self.max_epochs):
                with lsection(f"Epoch {epoch}:"):

                    # Train:
                    self.denoise_model.train()
                    self.deconv_model.train()
                    train_loss_value = 0
                    iteration = 0
                    for i, (input_images, target_images, mask_images) in enumerate(
                        train_data_loader
                    ):

                        lprint(
                            f"index: {i}, input shape:{input_images.shape}, target shape:{target_images.shape}"
                        )

                        # Adding training noise to input:
                        if self.input_noise > 0:
                            with torch.no_grad():
                                alpha = self.input_noise / (
                                    1 + (10000 * epoch / self.max_epochs)
                                )
                                lprint(f"Training noise level: {alpha}")
                                training_noise = alpha * torch.randn_like(input_images)
                                input_images += training_noise

                        # Clear gradients w.r.t. parameters
                        denoising_optimizer.zero_grad()

                        input_images_gpu = input_images.to(
                            self.device, non_blocking=True
                        )
                        target_images_gpu = target_images.to(
                            self.device, non_blocking=True
                        )
                        validation_mask_images_gpu = mask_images.to(
                            self.device, non_blocking=True
                        )

                        # Forward pass:
                        denoised_images_gpu = self.masked_denoise_model(
                            input_images_gpu
                        )

                        blind_spot_mask = self.masked_denoise_model.get_mask()

                        # denoise loss:
                        denoise_loss = loss_function(
                            denoised_images_gpu
                            * (1 - validation_mask_images_gpu)
                            * blind_spot_mask,
                            target_images_gpu
                            * (1 - validation_mask_images_gpu)
                            * blind_spot_mask,
                        ).mean()

                        # Back-propagation:
                        denoise_loss.backward(retain_graph=True)

                        # Updating parameters
                        denoising_optimizer.step()

                        # zero deconv gradients
                        deconv_optimizer.zero_grad()

                        # deconvolution:
                        deconvolved_images: torch.Tensor = self.deconv_model(
                            denoised_images_gpu
                        )
                        deconvolved_images.clamp_(0, 1)
                        blurred_deconvolved_images = self.psfconv(deconvolved_images)

                        # if epoch%20==0:
                        #     import napari
                        #     with napari.gui_qt():
                        #         viewer = napari.Viewer()
                        #         viewer.add_image(to_numpy(denoised_images_gpu), name='denoised_images_gpu* ~blind_spot_mask')
                        #         viewer.add_image(to_numpy(deconvolved_images), name='deconvolved_images')
                        #         viewer.add_image(to_numpy(blurred_deconvolved_images), name='blurred_deconvolved_images')

                        # Deconvolution loss:
                        deconv_loss = (
                            loss_function(
                                blurred_deconvolved_images
                                * (1 - validation_mask_images_gpu),
                                denoised_images_gpu * (1 - validation_mask_images_gpu),
                            ).mean()
                            + 0.000001 * torch.norm(deconvolved_images, 1).mean()
                        )

                        # Back-propagation:
                        deconv_loss.backward()

                        # Updating parameters
                        deconv_optimizer.step()
                        # self.deconv_model.post_optimisation()

                        # update training loss for whole image:
                        train_loss_value += denoise_loss.item() + deconv_loss.item()
                        iteration += 1

                    train_loss_value /= iteration
                    lprint(f"Training loss value: {train_loss_value}")

                    # Validate:
                    self.denoise_model.eval()
                    self.deconv_model.eval()
                    val_loss_value = 0
                    iteration = 0
                    for i, (input_images, target_images, mask_images) in enumerate(
                        val_data_loader
                    ):

                        input_images_gpu = input_images.to(
                            self.device, non_blocking=True
                        )
                        target_images_gpu = target_images.to(
                            self.device, non_blocking=True
                        )
                        validation_mask_images_gpu = mask_images.to(
                            self.device, non_blocking=True
                        )

                        with torch.no_grad():
                            # Forward pass:
                            denoised_images_gpu = self.masked_denoise_model(
                                input_images_gpu
                            )
                            deconvolved_images_gpu = self.deconv_model(
                                denoised_images_gpu
                            )
                            blured_images_gpu = self.psfconv(deconvolved_images_gpu)

                            # loss:
                            loss = loss_function(
                                blured_images_gpu * validation_mask_images_gpu,
                                target_images_gpu * validation_mask_images_gpu,
                            )

                            # Select validation voxels:
                            loss = loss.mean().cpu()

                            # update validation loss for whole image:
                            val_loss_value += loss.item()
                            iteration += 1

                    val_loss_value /= iteration
                    lprint(f"Validation loss value: {val_loss_value}")

                    # Learning rate schedule:
                    denoising_scheduler.step(val_loss_value)
                    deconv_scheduler.step(val_loss_value)

                    if val_loss_value < best_loss:
                        lprint(f"## New best val loss!")
                        if val_loss_value < best_loss - self.patience_epsilon:
                            lprint(f"## Good enough to reset patience!")
                            patience_counter = 0

                        best_loss = val_loss_value
                        from collections import OrderedDict

                        best_denoise_model_state_dict = OrderedDict(
                            {
                                k: v.to('cpu')
                                for k, v in self.denoise_model.state_dict().items()
                            }
                        )
                        best_deconv_model_state_dict = OrderedDict(
                            {
                                k: v.to('cpu')
                                for k, v in self.deconv_model.state_dict().items()
                            }
                        )

                    else:
                        # No improvement:
                        lprint(
                            f"No improvement of validation loss. patience = {patience_counter}/{self.patience} "
                        )
                        patience_counter += 1

                        if (
                            epoch % max(1, self.reload_best_model_period) == 0
                            and best_denoise_model_state_dict
                        ):  # epoch % 5 == 0 and
                            lprint(f"Reloading best model to date!")
                            self.denoise_model.load_state_dict(
                                best_denoise_model_state_dict
                            )
                            self.deconv_model.load_state_dict(
                                best_deconv_model_state_dict
                            )

                        if patience_counter > self.patience:
                            lprint(f"Early stopping!")
                            break

                    lprint(f"## Best loss value: {best_loss}")

                    if self._stop_training_flag:
                        lprint(f"Training interupted!")
                        break

        lprint(f"Reloading best model to date!")
        self.denoise_model.load_state_dict(best_denoise_model_state_dict)
        self.deconv_model.load_state_dict(best_deconv_model_state_dict)

    def _translate(self, input_image, batch_dims=None):
        """
            Internal method that translates an input image on the basis of the trained model.
        :param input_image: input image
        :param batch_dims: batch dimensions
        :return:
        """

        return self.denoise_and_deconvolve(input_image)

    def denoise(self, input_image):
        input_image = torch.Tensor(input_image[numpy.newaxis, numpy.newaxis, ...])
        input_image = input_image.to(self.device)
        inferred_image: torch.Tensor = self.denoise_model(input_image)
        inferred_image = inferred_image.detach().cpu().numpy()
        inferred_image = inferred_image.squeeze()
        return inferred_image

    def denoise_and_deconvolve(self, input_image):
        input_image = torch.Tensor(input_image[numpy.newaxis, numpy.newaxis, ...])
        input_image = input_image.to(self.device)
        inferred_image: torch.Tensor = self.deconv_model(
            self.denoise_model(input_image)
        ).clamp_(0, 1)
        inferred_image = inferred_image.detach().cpu().numpy()
        inferred_image = inferred_image.squeeze()
        return inferred_image
