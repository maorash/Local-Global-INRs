import os
from dataclasses import dataclass, field
from functools import partial
from typing import Optional, Union, List

import numpy as np
import pyrallis
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
    experiment_name: str = field(default="default_image_experiment")
    batch_size: int = field(default=1)
    lr: float = field(default=5e-4)
    num_epochs: int = field(default=1001)
    epochs_til_ckpt: int = field(default=1000)
    steps_til_summary: int = field(default=200)
    mode: str = field(default="mlp")
    activation_type: str = field(default="sine")
    hidden_features: int = field(default=256)
    image_size: int = field(default=512)
    downsample: Optional[List[int]] = field(default=None)
    checkpoint_path: Union[str, None] = field(default=None)
    debug: bool = field(default=False)


@dataclass
class LocalGlobalConfig(TrainingConfig):
    """Configuration specific to LocalGlobalBlock models."""
    agg_type: str = field(default="concat_and_fc")
    global_hidden_features: int = field(default=32)


@pyrallis.wrap()
def main(opt: LocalGlobalConfig):
    img_dataset = dataio.Camera()
    image_resolution = (opt.image_size, opt.image_size)
    coord_dataset = dataio.ImplicitSingle2DWrapper(img_dataset, sidelength=opt.image_size, compute_diff='all',
                                                   downsample=tuple(opt.downsample) if opt.downsample else None)
    dataloader = DataLoader(coord_dataset, shuffle=True, batch_size=opt.batch_size, pin_memory=True, num_workers=0)

    model = modules.INR(activation_type=opt.activation_type, mode=opt.mode,
                        sidelength=image_resolution, hidden_features=opt.hidden_features,
                        groups=None if not opt.downsample else np.prod(opt.downsample),
                        agg_type=opt.agg_type, global_hidden_features=opt.global_hidden_features)
    model.cuda()

    count_parameters = model.count_parameters()
    print(f"Parameters: {count_parameters}")

    if not opt.debug:
        root_path = os.path.join(opt.logging_root, opt.experiment_name)
        utils.cond_mkdir(root_path)

        # Define the loss
        loss_fn = partial(loss_functions.image_mse, None)
        summary_fn = partial(utils.write_image_summary, image_resolution, coord_dataset.coords_slices,
                             coord_dataset.gt_slices)

        if opt.use_wandb:
            wandb.init(project="Local-Global-SIRENs",
                       config={**vars(opt), **count_parameters},
                       name=opt.experiment_name,
                       sync_tensorboard=True,
                       tags=[opt.mode, f"DS_{opt.downsample}" if opt.downsample else 'DS_None'])

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
