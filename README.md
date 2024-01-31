# Local-Global SIRENs - Anonymous Submission

Code implementation for the paper "Croppable Implicit Neural Representations with Local-Global SIRENs".

## Abstract

Implicit Neural Representations (INRs) have peaked interest in recent years due to their ability to encode natural signals using neural networks. While INRs allow for useful applications such as interpolating new coordinates and signal compression, their black-box nature makes it difficult to modify them post-training. In this paper, we explore the idea of editable INRs, and specifically focus on the widely used cropping operation. To this end, we present Local-Global SIRENs -- a novel INR architecture that supports cropping by design. Local-Global SIRENs are based on combining local and global feature extraction for signal encoding. What makes their design unique, is the ability to effortlessly remove specific portions of an encoded signal, with a proportional weight decrease. This is achieved by eliminating the corresponding weights from the network, without the need for retraining. Furthermore, we demonstrate that this architecture enables the straightforward extension of previously encoded signals, highlighting their potential and flexibility.

## Table of Contents

- [Technological Overview](#technological-overview)
- [Installation](#installation)
- [Training](#training)
  - [Example - Image Encoding](#example---image-encoding)
  - [Example - Audio Encoding](#example---audio-encoding)
  - [Example - Video Encoding](#example---video-encoding)
- [Debug Outputs](#debug-outputs)
- [Decoding Videos](#decoding-videos)


## Technological Overview

* The code is based on the SIREN - "[SIREN: Implicit Neural Representations with Periodic Activation Functions](https://github.com/vsitzmann/siren)."
* For configuration management, we utilize Pyrallis. 
* WandB (Weights and Biases) is integrated into the code for experiment tracking and visualization. You can deactivate WandB by setting the `--use_wandb False` option.


## Installation
Run the following commands to set up the environment:
```bash
conda env create -f environment.yml
conda activate lgsirens
```

## Training
Refer to the appendix of the paper for the hyperparameters used for training.
- - The type of network is passed using `--mode`. The options are `lg` for Local-Global SIREN, `lc` for SIREN-per-Partition, and `mlp` for SIREN.
- Global hidden features are passed using `--global_hidden_features`
- Local hidden features are passed using `--hidden_features`. Note that the script expects the number of local hidden features to be a multiplied by the number of partitions. For example, if you want to use 14 local hidden features per partition for a total of 16*16=256 partitions, you should pass `--hidden_features 3584` (14*256=3584).
- The number of partitions is passed using `--downsample`. For example:
  - For an image encoding task, if you want to use 16*16=256 partitions, you should pass `--downsample [16,16]`.
  - For an audio encoding task, if you want to use 32 partitions, you should pass `--downsample [1,32,1]`.
  - For a video encoding task, if you want to use 5*16*16=1280 partitions, you should pass `--downsample [5,16,16]`.
- The number of overlapping coordinates in each dimension is passed using '--overlaps'. For example:
  - For a video encoding task, if you want to sample from 2 adjacent frames and from 1 pixel in each spatial dimension, you should pass `--overlaps [2,1,1]` (this is what we used in the paper).
- For video encoding tasks, the fraction of sampled pixels in each iteration is passed using `--sample_frac`. For example, if you want to sample 2% of the pixels in each iteration, you should pass `--sample_frac 0.02`.
- Experiment scripts are run from the experiments_scripts directory: 
```bash
cd experiment_scripts
```

### Example - Image Encoding
To train the model to encode an image, run the following command:
```bash
# From the experiment_scripts directory

# Local-Global SIREN
PYTHONPATH=../ python train_img.py --experiment_name test_image_lg --lr 5e-4 --num_epochs 1001 --hidden_features 3584 --epochs_til_ckpt 1000 --mode lg --global_hidden_features 84 --downsample [16,16]

# SIREN-per-Partition
PYTHONPATH=../ python train_img.py --experiment_name test_image_lc --lr 5e-4 --num_epochs 1001 --hidden_features 3840 --epochs_til_ckpt 1000 --mode lc --downsample [16,16]

# SIREN
PYTHONPATH=../ python train_img.py --experiment_name test_image_mlp --lr 5e-4 --num_epochs 1001 --hidden_features 256 --epochs_til_ckpt 1000 --mode mlp
```

### Example - Audio Encoding
To train the model to encode an audio file, run the following command:
```bash
# From the experiment_scripts directory

# Local-Global SIREN
PYTHONPATH=../ python train_img.py --experiment_name test_audio_lg --lr 1e-4 --num_epochs 1001 --hidden_features 1344 --epochs_til_ckpt 1000 --mode lg --global_hidden_features 72 --downsample [1,32,1]

# SIREN-per-Partition
PYTHONPATH=../ python train_img.py --experiment_name test_audio_lc --lr 1e-4 --num_epochs 1001 --hidden_features 1440 --epochs_til_ckpt 1000 --mode lc --downsample [1,32,1]

# SIREN
PYTHONPATH=../ python train_img.py --experiment_name test_audio_siren --lr 1e-4 --num_epochs 1001 --hidden_features 256 --epochs_til_ckpt 1000 --mode mlp
```

### Example - Video Encoding
To train the model to encode a video file, run the following command:
```bash
# From the experiment_scripts directory

# Local-Global SIREN
python train_video.py --experiment_name test_video_lg --num_epochs 5001 --epochs_til_ckpt 1000 --mode lg --dataset cat --downsample [5,8,8] --overlaps [1,2,2] --hidden_features 17600 --global_hidden_features 180 --sample_frac 0.02

# SIREN-per-Partition
python train_video.py --experiment_name test_video_lc --num_epochs 5001 --epochs_til_ckpt 1000 --mode lc --dataset cat --downsample [5,8,8] --overlaps [1,2,2] --hidden_features 17920 --sample_frac 0.02

# SIREN
python train_video.py --experiment_name test_video_siren --num_epochs 5001 --steps_til_summary 1000 --epochs_til_ckpt 5000 --mode mlp --dataset cat --hidden_features 1030
```

## Debug Outputs
During training, debug outputs are logged to the `logs/` directory by default (this can be changed using the `--logging_root` flag).
We log both the metrics and the visualizations using Tensorboard. When `--use_wandb` is passed, we also log the metrics to Weights and Biases.

## Decoding Videos
To decode a video using the trained model, use similar configuration as before, but pass the `--decode` flag and the `--checkpoint_path` flag. For example:
```bash
# From the experiment_scripts directory

# Local-Global SIREN
python decode_video.py --experiment_name test_video_lg --mode lg --dataset cat --downsample 5 8 8 --overlaps 2 1 1 --hidden_features 17600 --global_hidden_features 180 --decode --checkpoint_path model.pth
```
