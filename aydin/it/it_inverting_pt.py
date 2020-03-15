import math

import numpy
import torch
from torch import nn
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset

from aydin.it.it_base import ImageTranslatorBase
from aydin.it.pytorch.models.skipnet import SkipNet2D
from aydin.it.pytorch.optimisers.esadam import ESAdam
from aydin.util.log.log import lsection, lprint
from aydin.util.nd import extract_tiles

torch.manual_seed(0)
numpy.random.seed(0)
# torch.backends.cudnn.deterministic = True


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

        shape = input_image.shape
        ndim = input_image.ndim

        # tile size:
        tile_size = min(1024, min(shape))

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

        # Model
        self.model = SkipNet2D(1, 1)
        # lprint(f"Model: {self.model}")

        self.model = self.model.to(self.device)
        number_of_parameters = sum(
            p.numel() for p in self.model.trainable_parameters() if p.requires_grad
        )
        lprint(f"Number of trainable parameters in model: {number_of_parameters}")

        # Optimiser:
        optimiser_class = ESAdam
        lprint(f"Optimiser class: {optimiser_class}")
        lprint(f"Learning rate : {self.learning_rate}")
        optimizer = optimiser_class(
            self.model.trainable_parameters(),
            lr=self.learning_rate,
            weight_decay=0.01 * self.learning_rate,
        )
        lprint(f"Optimiser: {optimizer}")

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
            train_data_loader, val_data_loader, optimizer, loss_function, num_val_voxels
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

                if not is_training_data:
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
                if is_training_data:
                    return (input, target)
                else:
                    mask = self.mask_tiles[index, numpy.newaxis, ...]
                    return (input, target, mask)

        if is_training_data:
            input_image = input_image.copy()
            input_image[validation_voxels[0], validation_voxels[1]] = 0
            target_image = target_image.copy()
            target_image[validation_voxels[0], validation_voxels[1]] = 0

        if mode == 'grid':
            return _Dataset(input_image, target_image, tilesize)
        else:
            return None

    def _train_loop(
        self,
        train_data_loader,
        val_data_loader,
        optimizer,
        loss_function,
        num_val_voxels,
    ):

        # Scheduler:
        scheduler = ReduceLROnPlateau(
            optimizer, 'min', factor=0.99, verbose=True, patience=32
        )

        self.model.training = True

        best_loss = math.inf
        best_model_state_dict = None

        with lsection(f"Training loop:"):
            lprint(f"Maximum number of epochs: {self.max_epochs}")
            lprint(
                f"Training type: {'self-supervised' if self.self_supervised else 'supervised'}"
            )

            for epoch in range(self.max_epochs):
                with lsection(f"Epoch {epoch}:"):

                    # Train:
                    self.model.train()
                    train_loss_value = 0
                    iteration = 0
                    for i, (input_images, target_images) in enumerate(
                        train_data_loader
                    ):

                        lprint(
                            f"index: {i}, input shape:{input_images.shape}, target shape:{target_images.shape}"
                        )

                        # Adding training noise to input:
                        with torch.no_grad():
                            alpha = 1e-9
                            lprint(f"Training noise level: {alpha}")
                            training_noise = alpha * (
                                torch.rand_like(input_images) - 0.5
                            )
                            input_images += training_noise

                        # Clear gradients w.r.t. parameters
                        optimizer.zero_grad()

                        input_images_gpu = input_images.to(
                            self.device, non_blocking=True
                        )
                        target_images_gpu = target_images.to(
                            self.device, non_blocking=True
                        )

                        # Forward pass:
                        output_images_gpu = self.model(input_images_gpu)
                        # output_images = self.forward_model(output_images)

                        # loss:
                        loss = loss_function(
                            output_images_gpu, target_images_gpu
                        ).mean()

                        # Back-propagation:
                        loss.backward()

                        # update training loss for whole image:
                        train_loss_value += loss.item()
                        iteration += 1

                    train_loss_value /= iteration
                    lprint(f"Training loss value: {train_loss_value}")

                    # Updating parameters
                    self.model.pre_optimisation()
                    optimizer.step()
                    self.model.post_optimisation()

                    # Validate:
                    self.model.eval()
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
                        mask_images_gpu = mask_images.to(self.device, non_blocking=True)

                        with torch.no_grad():
                            # Forward pass:
                            output_images_gpu = self.model(input_images_gpu)

                            # loss:
                            loss = loss_function(
                                output_images_gpu * mask_images_gpu,
                                target_images_gpu * mask_images_gpu,
                            )

                            # Select validation voxels:
                            loss = loss.mean().cpu()

                            # update validation loss for whole image:
                            val_loss_value += (
                                loss.item()
                                * torch.numel(output_images_gpu)
                                / num_val_voxels
                            )
                            iteration += 1

                    val_loss_value /= iteration
                    lprint(f"Validation loss value: {val_loss_value}")

                    # Learning rate schedule:
                    scheduler.step(val_loss_value)

                    if val_loss_value < best_loss:
                        lprint(f"## New best loss!")
                        best_loss = val_loss_value
                        from collections import OrderedDict

                        best_model_state_dict = {
                            k: v.to('cpu') for k, v in self.model.state_dict().items()
                        }
                        best_model_state_dict = OrderedDict(best_model_state_dict)
                    else:
                        pass
                        # No improvement, reload best model weights to date:
                        if (
                            epoch % 64 == 0 and best_model_state_dict
                        ):  # epoch % 5 == 0 and
                            lprint(f"Reloading best model to date!")
                            self.model.load_state_dict(best_model_state_dict)
                        # optimizer.trigger_perturbation()

                    lprint(f"## Best loss value: {best_loss}")

                    if self._stop_training_flag:
                        lprint(f"Training interupted!")
                        break

        lprint(f"Reloading best model to date!")
        self.model.load_state_dict(best_model_state_dict)

    def _translate(self, input_image, batch_dims=None):
        """
            Internal method that translates an input image on the basis of the trained model.
        :param input_image: input image
        :param batch_dims: batch dimensions
        :return:
        """
        self.model.training = False
        input_image = torch.Tensor(input_image[numpy.newaxis, numpy.newaxis, ...])
        input_image = input_image.to(self.device)
        inferred_image: torch.Tensor = self.model(input_image)
        inferred_image = inferred_image.detach().cpu().numpy()
        inferred_image = inferred_image.squeeze()
        return inferred_image
