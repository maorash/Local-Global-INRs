# Local-Global INRs - Towards Croppable Implicit Neural Representations

* Extended qualitative results, including audio, video and cropping are available at:
  * [https://sites.google.com/view/local-global-inrs](https://sites.google.com/view/local-global-inrs).


The supplementary material contains the code implementation for the paper *"Towards Croppable Implicit Neural Representations"*.

## Opening Notes
We extend our gratitude to the anonymous reviewers who are dedicating their valuable time to review our paper and code. 
Your constructive feedback plays an indispensable role in enhancing the quality of our work. 

Best regards,

The Local-Global INRs team

## Abstract

Implicit Neural Representations (INRs) have peaked interest in recent years due to their ability to encode natural signals using neural networks. While INRs allow for useful applications such as interpolating new coordinates and signal compression, their black-box nature makes it difficult to modify them post-training. In this paper we explore the idea of editable INRs, and specifically focus on the widely used cropping operation. To this end, we present Local-Global SIRENs -- a novel INR architecture that supports cropping by design. Local-Global SIRENs are based on combining local and global feature extraction for signal encoding. What makes their design unique is the ability to effortlessly remove specific portions of an encoded signal, with a proportional weight decrease. This is achieved by eliminating the corresponding weights from the network, without the need for retraining. We further show how this architecture can be used to support the straightforward extension of previously encoded signals. Beyond signal editing, we examine how the Local-Global approach can accelerate training, enhance encoding on various signals, improve downstream performance, and be applied to modern INRs such as INCODE, highlighting its potential and flexibility.

## Table of Contents

- [Technological Overview and Sources](#technological-overview-and-sources)
- [Installation](#installation)
- [Encoding Images, Audio and Video](#training-on-images-audio-and-videos)
  - [Example - Image Encoding](#example---image-encoding)
  - [Example - Audio Encoding](#example---audio-encoding)
  - [Example - Video Encoding](#example---video-encoding)
- [Encoding Images with Automatic Partitioning](#encoding-images-with-automatic-partitioning)
- [Debug Outputs](#debug-outputs)
- [Decoding and Cropping Videos](#decoding-and-cropping-videos)


## Technological Overview and Sources

* Most of our codebase is based on SIREN:
  * [SIREN: Implicit Neural Representations with Periodic Activation Functions](https://github.com/vsitzmann/siren)
  * Released under MIT License
* Audio and video data samples under the `data` directory are taken from the official SIREN repository.
* INCODE experiments are based on the official implementation:
  * [INCODE: Implicit Neural Conditioning with Prior Knowledge Embeddings(https://github.com/xmindflow/INCODE)
  * Released under MIT License
* The DIV2K images used in the experiments in the paper are under `data/DIV2K`
  * `data/DIV2K/DIV2K_subset` contains the randomly selected 25 images, used in the image encoding experiments.
  * `data/DIV2K/denoising` contains the image used in the denoising experiment.
  * `data/DIV2K/superresolution` contains the image used in the super-resolution experiment.
  * The full DIV2K dataset is available at [DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/)
  * The license agreement for DIV2K is mentioned in their website (academic use only).
* The image used for CT reconstruction is `data/img_377_ct_reconstruction.png`
  * The image is taken from the [Kaggle Lung Nodule Analysis dataset](https://luna16.grand-challenge.org/)
* The shape used for 3D encoding is the Lucy dataset
  * The shape is taken from [the Stanford 3D Scanning Repository](https://graphics.stanford.edu/data/3Dscanrep/)
* For configuration management, we use `pyrallis`. 
* WandB (Weights and Biases) is integrated into the code for experiment tracking and visualization. You can deactivate WandB by setting the `--use_wandb False` option.


## Installation
Run the following commands to set up the environment:
```bash
conda env create -f environment.yml
conda activate lgsirens
```

## Training on Images, Audio and Videos

> Refer to the appendix of the paper for the hyperparameters used for training.

- The type of network is passed using `--mode`. The options are `lg` for Local-Global SIREN, `lc` for SIREN-per-Partition, and `mlp` for SIREN.
- Global hidden features are passed using `--global_hidden_features`
- Local hidden features are passed using `--hidden_features`.
  - Note that the script expects the number of local hidden features to be a multiplied by the number of partitions. For example, if you want to use 14 local hidden features per partition for a total of 16\*16=256 partitions, you should pass `--hidden_features 3584` (14\*256=3584).
- The number of partitions is passed using `--downsample`. For example:
  - For an image encoding task, if you want to use 16\*16=256 partitions, you should pass `--downsample [16,16]`.
  - For an audio encoding task, if you want to use 32 partitions, you should pass `--downsample [1,32,1]`.
  - For a video encoding task, if you want to use 5\*16\*16=1280 partitions, you should pass `--downsample [5,16,16]`.
- The number of overlapping coordinates in each dimension is passed using '--overlaps'. For example:
  - For a video encoding task, if you want to sample from 2 adjacent frames and from 1 pixel in each spatial dimension, you should pass `--overlaps [2,1,1]` (this is what we used in the paper).
- For video encoding tasks, the fraction of sampled pixels in each iteration is passed using `--sample_frac`. For example, if you want to sample 2% of the pixels in each iteration, you should pass `--sample_frac 0.02`.
- Experiment scripts are run from the experiments_scripts directory: 

> We present examples for image, audio, and video encoding tasks. The examples are based on the hyperparameters used in the paper.

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

## Encoding Images with Automatic Partitioning
There are three parameters that control the automatic partitioning logic:
* First and foremost, the `--partition_size` argument sets the desired partition size
* The `--auto_total_number_of_parameters` argument sets the required total number of parameters in the network
  * It is set by default to 200k, and might deviate by roughly 5-10%
  * This is equivalent to an MLP with 3 hidden layers and 256 hidden units
* The `--auto_global_weights_factor' argument sets the ratio of global weights to local weights
  * It is set by default to 10%, and might deviate by roughly 5-10%
  * We use the same ratio for the global weights when encoding images throughout the paper


### Example
To train the model to encode an image with automatic partitioning on a DIV2K image, downsampled by 4 before training, run the following command:

```bash
# From the experiment_scripts directory

# Local-Global SIREN
PYTHONPATH=../ python train_img.py --experiment_name test_image_lg --dataset_path <PATH_TO_DIV2K_IMAGE> --image_resolution_factor 4 --num_epochs 2001 --steps_til_summary 200 --epochs_til_ckpt 1000 --mode lg --partition_size [32,32]
```

## Debug Outputs
During training, debug outputs are logged to the `logs/` directory by default (this can be changed using the `--logging_root` flag).
We log both the metrics and the visualizations using Tensorboard. When `--use_wandb` is passed, we also log the metrics to Weights and Biases.

## Decoding and Cropping Videos
To decode a video using the trained model, use similar configuration as before, but pass the `--decode` flag and the `--checkpoint_path` flag. For example:
```bash
# From the experiment_scripts directory

# Local-Global SIREN
python decode_video.py --experiment_name test_video_lg --mode lg --dataset cat --downsample 5 8 8 --overlaps 2 1 1 --hidden_features 17600 --global_hidden_features 180 --decode --checkpoint_path model.pth
```

To crop specific partitions of the video, use any one (or more) of the following flags:
- `--crop_entire_dim_values`: A list of three lists. Crop partitions based on indices in specific dimensions, across all the signal. For example:
  - To crop the entire spatial border of the video (across all frames), assuming the video was downsampled by a factor of 8 in each spatial dimension, pass `--crop_entire_dim_values [[],[0,7],[0,7]]`.
  - To crop the second and third partitions in the temporal dimension (across all spatial locations), pass `--crop_entire_dim_values [[1,2],[],[]]`.
- '--crop_partition_indices': A list of lists of size three. Crop specific partitions based on their indices (in this case, each partition is indexed by three coordinates). For example:
  - To crop partition at the start of the video, in the bottom right corner (assuming the video was downsampled by a factor of 8 in each spatial dimension), pass `--crop_partition_indices [[0,7,7]]`.
