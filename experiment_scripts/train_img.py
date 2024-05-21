import os
from dataclasses import dataclass, field
from functools import partial
from typing import Optional, Union, List

import math
import numpy as np
import pyrallis
import wandb
from torch.utils.data import DataLoader

import dataio
import loss_functions
import modules
import training
import utils
from compute_partitions import compute_hidden_dims


@dataclass
class TrainingConfig:
    dataset_path: str = field(default="cameraman") # Either "cameraman" or a path to an image
    image_channels: int = field(default=3)
    image_resolution_factor: float = field(default=1)

    use_wandb: bool = field(default=True)
    logging_root: str = field(default="./logs")
    experiment_name: str = field(default="default_image_experiment")
    batch_size: int = field(default=1)
    lr: float = field(default=5e-4)
    num_epochs: int = field(default=2001)
    epochs_til_ckpt: int = field(default=1000)
    steps_til_summary: int = field(default=200)

    mode: str = field(default="mlp")
    activation_type: str = field(default="sine")
    hidden_features: int = field(default=256)
    num_hidden_layers: int = field(default=3)
    downsample: Optional[List[int]] = field(default=None)
    checkpoint_path: Union[str, None] = field(default=None)
    debug: bool = field(default=False)

    # Automatic partitioning
    partition_size: Optional[List[int]] = field(default=None)
    auto_total_number_of_parameters: int = field(default=200_000)
    auto_global_weights_factor: float = field(default=0.1)


@dataclass
class LocalGlobalConfig(TrainingConfig):
    """Configuration specific to LocalGlobalBlock models."""
    agg_type: str = field(default="concat_and_fc")
    global_hidden_features: int = field(default=32)


@pyrallis.wrap()
def main(opt: LocalGlobalConfig):
    if opt.dataset_path == "cameraman":
        img_dataset = dataio.Camera()
        opt.image_channels = 1
    else:
        img_dataset = dataio.ImageFile(opt.dataset_path)

    H, W, opt.downsample, opt.hidden_features, opt.global_hidden_features = resize_and_select_partitioning(opt, img_dataset)
    image_resolution = (H, W)

    coord_dataset = dataio.ImplicitSingle2DWrapper(img_dataset, sidelength=image_resolution,
                                                   compute_diff='all' if opt.image_channels == 1 else None,
                                                   downsample=tuple(opt.downsample) if opt.downsample else None)
    dataloader = DataLoader(coord_dataset, shuffle=True, batch_size=opt.batch_size, pin_memory=True, num_workers=0)

    model = modules.INR(out_features=opt.image_channels,
                        activation_type=opt.activation_type, mode=opt.mode,
                        sidelength=image_resolution, hidden_features=opt.hidden_features,
                        groups=None if not opt.downsample else np.prod(opt.downsample),
                        agg_type=opt.agg_type, global_hidden_features=opt.global_hidden_features,
                        num_hidden_layers=opt.num_hidden_layers)
    model.cuda()

    count_parameters = model.count_parameters()
    print(f"Parameters: {count_parameters}")

    if not opt.debug:
        root_path = os.path.join(opt.logging_root, opt.experiment_name)
        utils.cond_mkdir(root_path)

        # Define the loss
        loss_fn = partial(loss_functions.image_mse, None)
        summary_fn = partial(utils.write_image_summary, image_resolution, coord_dataset.coords_slices,
                             coord_dataset.gt_slices, show_grads=False)

        if opt.use_wandb:
            wandb.init(project="Local-Global-SIRENs",
                       config={**vars(opt), **count_parameters},
                       name=opt.experiment_name,
                       sync_tensorboard=True,
                       tags=[opt.mode, f"DS_{opt.downsample}" if opt.downsample else 'DS_None'])

        training_time, summary_values = training.train(model=model, train_dataloader=dataloader, epochs=opt.num_epochs,
                                                       lr=opt.lr,
                                                       steps_til_summary=opt.steps_til_summary,
                                                       epochs_til_checkpoint=opt.epochs_til_ckpt,
                                                       model_dir=root_path, loss_fn=loss_fn, summary_fn=summary_fn,
                                                       summary_values_compare_fn=lambda x, y: x[0] > y[0])

        psnrs, ssims = zip(*summary_values)

        if opt.use_wandb:
            wandb.run.summary["training_time"] = training_time
            wandb.finish()
    else:
        training.train_minimum(model=model, dataloader=dataloader, lr=opt.lr, steps=opt.num_epochs)


def resize_and_select_partitioning(opt: LocalGlobalConfig, img_dataset: dataio.ImageFile,
                                   align_dimensions_to_partitions=True):
    H, W = img_dataset.img.height, img_dataset.img.width
    print(f"Image size before resizing: {H, W}")

    downsample = tuple(opt.downsample) if opt.downsample is not None else None

    # If user requested a partition size, compute the partitioning factors
    if opt.partition_size is not None:
        downsample = (math.ceil((max(H, W) // opt.image_resolution_factor) / opt.partition_size[0]),
                      math.ceil((min(H, W) // opt.image_resolution_factor) / opt.partition_size[1]))
        if W > H:
            downsample = (downsample[1], downsample[0]) if downsample is not None else None

    num_groups = np.prod(downsample)
    if 'lg' in opt.mode:
        if opt.partition_size is not None:
            # Automatic partitioning
            local_hidden_dim, global_hidden_dim = compute_hidden_dims(total_param_number=opt.auto_total_number_of_parameters,
                                                                      global_weights_factor=opt.auto_global_weights_factor,
                                                                      in_dim=2,
                                                                      output_dim=opt.image_channels,
                                                                      hidden_layers=opt.num_hidden_layers,
                                                                      groups=num_groups)
            local_hidden_dim *= num_groups
        else:
            # Hard-coded partitioning
            local_hidden_dim, global_hidden_dim = opt.hidden_features, opt.global_hidden_features
    elif 'lc' in opt.mode:
        if opt.partition_size is not None:
            # Automatic partitioning
            local_hidden_dim, global_hidden_dim = compute_hidden_dims(total_param_number=opt.auto_total_number_of_parameters,
                                                                      global_weights_factor=0,
                                                                      in_dim=2,
                                                                      output_dim=opt.image_channels,
                                                                      hidden_layers=opt.num_hidden_layers,
                                                                      groups=num_groups)
            global_hidden_dim = None
            local_hidden_dim *= num_groups
        else:
            # Hard-coded partitioning
            local_hidden_dim, global_hidden_dim = opt.hidden_features, None
    else:
        # No partitioning
        local_hidden_dim, global_hidden_dim = opt.hidden_features, None

    print(f"Selected partition factors: {downsample}")
    print(f"Selected local hidden dim: {local_hidden_dim}")
    print(f"Selected global hidden dim: {global_hidden_dim}")

    if align_dimensions_to_partitions:
        # Resize to the nearest multiple of the partition factors
        if (H // opt.image_resolution_factor % downsample[0]) < downsample[0] // 2:
            H = (H // opt.image_resolution_factor) - (H // opt.image_resolution_factor % downsample[0])
        else:
            H = (H // opt.image_resolution_factor) + (downsample[0] - (H // opt.image_resolution_factor % downsample[0]))

        if (W // opt.image_resolution_factor % downsample[1]) < downsample[1] // 2:
            W = (W // opt.image_resolution_factor) - (W // opt.image_resolution_factor % downsample[1])
        else:
            W = (W // opt.image_resolution_factor) + (downsample[1] - (W // opt.image_resolution_factor % downsample[1]))

    H, W = int(H), int(W)
    print(f"Image size after resizing: {H, W}")

    return H, W, downsample, local_hidden_dim, global_hidden_dim

main()
