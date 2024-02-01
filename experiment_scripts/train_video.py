import os
from dataclasses import dataclass, field
from functools import partial
from typing import Optional, Union, List, Tuple

import numpy as np
import pyrallis
import skvideo.datasets
import torch
import wandb
from torch.utils.data import DataLoader

import dataio
import loss_functions
import modules
import training
import utils


@dataclass
class TrainingConfig:
    use_wandb: bool = field(default=True)
    logging_root: str = field(default="./logs")
    experiment_name: str = field(default="default_video_experiment")
    batch_size: int = field(default=1)
    lr: float = field(default=1e-4)
    num_epochs: int = field(default=5001)
    epochs_til_ckpt: int = field(default=5000)
    steps_til_summary: int = field(default=1000)
    dataset: str = field(default="bikes")
    sample_frac: float = field(default=38e-4)
    mode: str = field(default="mlp")
    activation_type: str = field(default="sine")
    hidden_features: int = field(default=1030)
    downsample: Optional[List[int]] = field(default=None)
    overlaps: Optional[List[int]] = field(default=None)
    checkpoint_path: Union[str, None] = field(default=None)
    debug: bool = field(default=False)
    decode: bool = field(default=False)
    crop_entire_dim_values: List[List[int]] = field(default=None)
    crop_partition_indices: List[Tuple[int, int, int]] = field(default=None)


@dataclass
class LocalGlobalConfig(TrainingConfig):
    """Configuration specific to LocalGlobalBlock models."""
    agg_type: str = field(default="concat_and_fc")
    global_hidden_features: int = field(default=32)


@pyrallis.wrap()
def main(opt: LocalGlobalConfig):
    if opt.dataset == 'cat':
        video_path = '../data/cat_video.mp4'
    elif opt.dataset == 'bikes':
        video_path = skvideo.datasets.bikes()
    elif opt.dataset == 'bunny':
        video_path = skvideo.datasets.bigbuckbunny()
    else:
        print("unsupported dataset")
        return

    vid_dataset = dataio.Video(video_path)
    coord_dataset = dataio.Implicit3DWrapper(vid_dataset, sidelength=vid_dataset.shape, sample_fraction=opt.sample_frac,
                                             downsample=tuple(opt.downsample) if opt.downsample else None,
                                             overlaps=tuple(opt.overlaps) if opt.overlaps else None)
    dataloader = DataLoader(coord_dataset, shuffle=True, batch_size=opt.batch_size, pin_memory=True, num_workers=0)

    model = modules.INR(activation_type=opt.activation_type, mode=opt.mode,
                        hidden_features=opt.hidden_features, in_features=3, out_features=vid_dataset.channels,
                        groups=None if not opt.downsample else np.prod(opt.downsample),
                        agg_type=opt.agg_type, global_hidden_features=opt.global_hidden_features, num_hidden_layers=3)

    model.cuda()

    count_parameters = model.count_parameters()
    print(f"Parameters: {count_parameters}")

    experiment_name = f"{opt.experiment_name}{opt.downsample if opt.downsample else ''}{opt.overlaps if opt.overlaps else ''}"

    if opt.decode:
        if opt.checkpoint_path is None:
            print("Must specify checkpoint path")
            return
        model.load_state_dict(torch.load(opt.checkpoint_path))

        if opt.mode in ['lc', 'lg']:
            cropped_dim_values = opt.crop_entire_dim_values if opt.crop_entire_dim_values is not None else [[], [], []]
            cropped_partition_indices = opt.crop_partition_indices if opt.crop_partition_indices is not None else []
            video_shape = coord_dataset.dataset.shape
            partition_deltas = [video_shape[i] // opt.downsample[i] for i in range(len(video_shape))]
            for i in range(len(cropped_dim_values)):
                cropped_dim_values[i] = [j * partition_deltas[i] for j in cropped_dim_values[i]]
            for i in range(len(cropped_partition_indices)):
                cropped_partition_indices[i] = tuple(np.array(partition_deltas) * np.array(cropped_partition_indices[i]))

            flattened_partition_indices = [i for i, s in enumerate(coord_dataset.coords_slices) if (
                        s[0].start in cropped_dim_values[0] or s[1].start in cropped_dim_values[1] or s[2].start in cropped_dim_values[2])]
            for partition_index in cropped_partition_indices:
                for i, s in enumerate(coord_dataset.coords_slices):
                    if s[0].start == partition_index[0] and s[1].start == partition_index[1] and s[2].start == partition_index[2]:
                        flattened_partition_indices.append(i)

            model.crop(flattened_partition_indices)
            print(f"Parameters after crop: {model.count_parameters(cropped_partition_indices=flattened_partition_indices)}")

        video = utils.decode_video(model, coord_dataset, opt.mode in ['lc', 'lg'])
        dataio.Video.write_video(video, f"{experiment_name}.mp4")
        return

    if not opt.debug:
        root_path = os.path.join(opt.logging_root, opt.experiment_name)
        utils.cond_mkdir(root_path)

        # Define the loss
        loss_fn = partial(loss_functions.image_mse, None)
        summary_fn = partial(utils.write_video_summary, coord_dataset, opt.mode in ['lc', 'lg'])

        if opt.use_wandb:
            wandb.init(project="Local-Global-SIRENs",
                       config={**vars(opt), **count_parameters},
                       name=experiment_name,
                       sync_tensorboard=True,
                       tags=[opt.mode,
                             f"DS_{opt.downsample}" if opt.downsample else 'DS_None',
                             f"OL_{opt.overlaps}" if opt.overlaps else 'OL_None'])

        training_time = training.train(model=model, train_dataloader=dataloader, epochs=opt.num_epochs, lr=opt.lr,
                                       steps_til_summary=opt.steps_til_summary,
                                       epochs_til_checkpoint=opt.epochs_til_ckpt,
                                       model_dir=root_path, loss_fn=loss_fn, summary_fn=summary_fn)

        if opt.use_wandb:
            wandb.run.summary["training_time"] = training_time
            wandb.finish()
    else:
        training.train_minimum(model=model, dataloader=dataloader, lr=opt.lr, steps=opt.num_epochs)


main()
