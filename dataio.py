from itertools import product

import math
import matplotlib.colors as colors
import numpy as np
import scipy.io.wavfile as wavfile
import scipy.ndimage
import scipy.special
import skimage
import skimage.filters
import torchvision

np.float = np.float64
np.int = np.int_
import skvideo

# Set ffmpeg path
skvideo.setFFmpegPath('../../ffmpeg-6.1-amd64-static/')
import skvideo.io
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import Resize, Compose, ToTensor


def get_mgrid(sidelen, dim=2):
    '''Generates a flattened grid of (x,y,...) coordinates in a range of -1 to 1.'''
    if isinstance(sidelen, int):
        sidelen = dim * (sidelen,)

    if dim == 2:
        pixel_coords = np.stack(np.mgrid[:sidelen[0], :sidelen[1]], axis=-1)[None, ...].astype(np.float32)
        pixel_coords[0, :, :, 0] = pixel_coords[0, :, :, 0] / (sidelen[0] - 1)
        pixel_coords[0, :, :, 1] = pixel_coords[0, :, :, 1] / (sidelen[1] - 1)
    elif dim == 3:
        pixel_coords = np.stack(np.mgrid[:sidelen[0], :sidelen[1], :sidelen[2]], axis=-1)[None, ...].astype(np.float32)
        pixel_coords[..., 0] = pixel_coords[..., 0] / max(sidelen[0] - 1, 1)
        pixel_coords[..., 1] = pixel_coords[..., 1] / (sidelen[1] - 1)
        pixel_coords[..., 2] = pixel_coords[..., 2] / (sidelen[2] - 1)
    else:
        raise NotImplementedError('Not implemented for dim=%d' % dim)

    pixel_coords -= 0.5
    pixel_coords *= 2.
    pixel_coords = torch.Tensor(pixel_coords).view(-1, dim)
    return pixel_coords


def lin2img(tensor, image_resolution=None, slices=None):
    if slices is None:
        batch_size, num_samples, channels = tensor.shape
    else:
        batch_size, groups, group_samples, channels = tensor.shape
        num_samples = groups * group_samples

    if image_resolution is None:
        width = np.sqrt(num_samples).astype(int)
        height = width
    else:
        height = image_resolution[0]
        width = image_resolution[1]

    if slices is not None:
        reconstructed = torch.zeros((batch_size, height, width, channels))
        for batch_index in range(batch_size):
            for im_slice, im_block in zip(slices, tensor[batch_index]):
                reconstructed[batch_index][im_slice] = im_block.reshape(reconstructed[batch_index][im_slice].shape)
        return reconstructed.permute(0, 3, 1, 2)
    else:
        return tensor.permute(0, 2, 1).view(batch_size, channels, height, width)


def lin2vid(tensor, video_resolution, slices=None):
    if slices is not None:
        reconstructed = torch.zeros(video_resolution)
        for vid_slice, vid_block in zip(slices, tensor):
            reconstructed[vid_slice] = vid_block.reshape(reconstructed[vid_slice].shape)
        return reconstructed
    else:
        return tensor.reshape(video_resolution)


def grads2img(gradients):
    mG = gradients.detach().squeeze(0).permute(-2, -1, -3).cpu()
    # assumes mG is [row,cols,2]
    nRows = mG.shape[0]
    nCols = mG.shape[1]
    mGr = mG[:, :, 0]
    mGc = mG[:, :, 1]
    mGa = np.arctan2(mGc, mGr)
    mGm = np.hypot(mGc, mGr)
    mGhsv = np.zeros((nRows, nCols, 3), dtype=np.float32)
    mGhsv[:, :, 0] = (mGa + math.pi) / (2. * math.pi)
    mGhsv[:, :, 1] = 1.

    nPerMin = np.percentile(mGm, 5)
    nPerMax = np.percentile(mGm, 95)
    mGm = (mGm - nPerMin) / (nPerMax - nPerMin)
    mGm = np.clip(mGm, 0, 1)

    mGhsv[:, :, 2] = mGm
    mGrgb = colors.hsv_to_rgb(mGhsv)
    return torch.from_numpy(mGrgb).permute(2, 0, 1)


def rescale_img(x, mode='scale', perc=None, tmax=1.0, tmin=0.0):
    if mode == 'scale':
        if perc is None:
            xmax = torch.max(x)
            xmin = torch.min(x)
        else:
            xmin = np.percentile(x.detach().cpu().numpy(), perc)
            xmax = np.percentile(x.detach().cpu().numpy(), 100 - perc)
            x = torch.clamp(x, xmin, xmax)
        if xmin == xmax:
            return 0.5 * torch.ones_like(x) * (tmax - tmin) + tmin
        x = ((x - xmin) / (xmax - xmin)) * (tmax - tmin) + tmin
    elif mode == 'clamp':
        x = torch.clamp(x, 0, 1)
    return x


def to_uint8(x):
    return (255. * x).astype(np.uint8)


class Video(Dataset):
    def __init__(self, path_to_video):
        super().__init__()
        if 'npy' in path_to_video:
            self.vid = np.load(path_to_video)
        elif 'mp4' in path_to_video:
            self.vid = skvideo.io.vread(path_to_video).astype(np.single) / 255.

        self.shape = self.vid.shape[:-1]
        self.channels = self.vid.shape[-1]

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return self.vid

    @staticmethod
    def write_video(vid, path):
        vid = vid * 255
        skvideo.io.vwrite(path, vid.astype(np.uint8))
        skvideo.io.vwrite(path, vid)


class Camera(Dataset):
    def __init__(self):
        super().__init__()
        self.img = Image.fromarray(skimage.data.camera())
        self.img_channels = 1

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return self.img


class ImagePil(Dataset):
    def __init__(self, img):
        super().__init__()
        self.img = img
        self.img_channels = len(self.img.mode)

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return self.img


class ImageFile(Dataset):
    def __init__(self, filename):
        super().__init__()
        self.img = Image.open(filename)
        self.img_channels = len(self.img.mode)

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return self.img


class ImageFileFromArray(Dataset):
    def __init__(self, img: np.array):
        super().__init__()
        self.img_channels = img.shape[-1] if len(img.shape) == 3 else 1
        self.img = img

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return self.img


class ImplicitAudioWrapper(torch.utils.data.Dataset):
    def __init__(self, dataset, downsample=None):
        self.dataset = dataset
        if len(dataset) != 1:
            raise ValueError("Wrapper only supports a single 1D object")
        self.grid = np.linspace(start=-100, stop=100, num=self.dataset.file_length)
        self.grid = self.grid.astype(np.float32)
        self.grid = torch.Tensor(self.grid).view(-1, 1)

        rate, data = self.dataset[0]
        scale = np.max(np.abs(data))
        data = (data / scale)
        data = torch.Tensor(data).view(-1, 1)

        self.in_dict = {'idx': 0, 'coords': self.grid}
        self.gt_dict = {'func': data, 'rate': rate, 'scale': scale, 'raw_func': self.dataset.data}

        if downsample is None:
            self.coords_slices, self.gt_slices = None, None
        else:
            self.in_dict['coords'], self.coords_slices = partition_signal(self.in_dict['coords'],
                                                                          (1, self.dataset.file_length), downsample)
            self.gt_dict['func'], self.gt_slices = partition_signal(self.gt_dict['func'],
                                                                    (1, self.dataset.file_length), downsample)

    def get_num_samples(self):
        return self.grid.shape[0]

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return self.in_dict, self.gt_dict


class AudioFile(Dataset):
    def __init__(self, filename):
        super().__init__()
        self.rate, self.data = wavfile.read(filename)
        if len(self.data.shape) > 1 and self.data.shape[1] == 2:
            self.data = np.mean(self.data, axis=1)
        self.data = self.data.astype(np.float32)
        self.file_length = len(self.data)
        print("Rate: %d" % self.rate)

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return self.rate, self.data


class Implicit2DWrapper(torch.utils.data.Dataset):
    def __init__(self, dataset, sidelength=None, compute_diff=None, skip_transforms=False):
        if isinstance(sidelength, int):
            sidelength = (sidelength, sidelength)
        self.sidelength = sidelength
        self.skip_transforms = skip_transforms

        self.size_transform = Compose([
            Resize(sidelength)
        ])
        self.tensor_transform = Compose([
            ToTensor()
        ])

        self.compute_diff = compute_diff
        self.dataset = dataset
        self.mgrid = get_mgrid(sidelength)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if self.skip_transforms:
            raw_image = self.tensor_transform(self.dataset[idx])
        else:
            raw_image = self.tensor_transform(self.size_transform(self.dataset[idx]))
        img = (raw_image - 0.5) / 0.5

        if self.compute_diff == 'gradients':
            img *= 1e1
            gradx = scipy.ndimage.sobel(img.numpy(), axis=1).squeeze(0)[..., None]
            grady = scipy.ndimage.sobel(img.numpy(), axis=2).squeeze(0)[..., None]
        elif self.compute_diff == 'laplacian':
            img *= 1e4
            laplace = scipy.ndimage.laplace(img.numpy()).squeeze(0)[..., None]
        elif self.compute_diff == 'all':
            gradx = scipy.ndimage.sobel(img.numpy(), axis=1).squeeze(0)[..., None]
            grady = scipy.ndimage.sobel(img.numpy(), axis=2).squeeze(0)[..., None]
            laplace = scipy.ndimage.laplace(img.numpy()).squeeze(0)[..., None]

        img = img.permute(1, 2, 0).view(-1, self.dataset.img_channels)

        in_dict = {'idx': idx, 'coords': self.mgrid}
        gt_dict = {'img': img, 'raw_img': np.array(raw_image)}

        if self.compute_diff == 'gradients':
            gradients = torch.cat((torch.from_numpy(gradx).reshape(-1, 1),
                                   torch.from_numpy(grady).reshape(-1, 1)),
                                  dim=-1)
            gt_dict.update({'gradients': gradients})

        elif self.compute_diff == 'laplacian':
            gt_dict.update({'laplace': torch.from_numpy(laplace).view(-1, 1)})

        elif self.compute_diff == 'all':
            gradients = torch.cat((torch.from_numpy(gradx).reshape(-1, 1),
                                   torch.from_numpy(grady).reshape(-1, 1)),
                                  dim=-1)
            gt_dict.update({'gradients': gradients})
            gt_dict.update({'laplace': torch.from_numpy(laplace).view(-1, 1)})

        return in_dict, gt_dict

    def get_item_small(self, idx):
        raw_image = self.size_transform(self.dataset[idx])
        img = self.tensor_transform(raw_image)
        spatial_img = img.clone()
        img = img.permute(1, 2, 0).view(-1, self.dataset.img_channels)

        gt_dict = {'img': img, 'raw_img': np.array(raw_image)}

        return spatial_img, img, gt_dict


class ImplicitSingle2DWrapper(torch.utils.data.Dataset):
    def __init__(self, dataset, sidelength=None, compute_diff=None, downsample=None, skip_transforms=False, n_samples=None, overlaps=None):
        if len(dataset) != 1:
            raise ValueError("Wrapper only supports a single 2D object")

        self.dataset = dataset
        self.dataset_wrapper = Implicit2DWrapper(dataset, sidelength, compute_diff, skip_transforms)
        self.in_dict, self.gt_dict = self.dataset_wrapper.__getitem__(0)
        self.downsample = downsample
        self.num_groups = np.prod(downsample) if downsample is not None else 1
        self.n_samples = n_samples // self.num_groups if n_samples is not None else None

        if downsample is None:
            self.coords_slices, self.gt_slices = None, None
            self.in_dict['coords_overlap'] = self.in_dict['coords']
            self.gt_dict['img_overlap'] = self.gt_dict['img']
        else:
            coords = self.in_dict['coords']
            img = self.gt_dict['img']

            self.in_dict['coords'], self.coords_slices = partition_signal(coords,
                                                                          self.dataset_wrapper.sidelength, downsample)
            self.gt_dict['img'], self.gt_slices = partition_signal(img,
                                                                   self.dataset_wrapper.sidelength, downsample)

            self.in_dict['coords_overlap'], self.coords_slices_overlaps = partition_signal(coords,
                                                                                           self.dataset_wrapper.sidelength,
                                                                                           downsample,
                                                                                           overlaps=overlaps)
            self.gt_dict['img_overlap'], self.gt_slices_overlaps = partition_signal(img,
                                                                                    self.dataset_wrapper.sidelength,
                                                                                    downsample,
                                                                                    overlaps=overlaps)

        self.num_pixels = self.dataset_wrapper.sidelength[0] * self.dataset_wrapper.sidelength[1]

    def __len__(self):
        return len(self.dataset_wrapper)

    def __getitem__(self, idx):
        if self.n_samples is not None:
            if self.downsample is None:
                selected_indices = torch.randperm(self.num_pixels)[:self.n_samples]
                img = self.gt_dict['img'][selected_indices, :]
                coords = self.in_dict['coords'][selected_indices, :]
            else:
                selected_indices = torch.randint(0, self.gt_dict['img_overlap'].shape[1], (self.num_groups,
                                                                                           self.n_samples))
                img = torch.gather(self.gt_dict['img_overlap'], dim=1,
                                   index=selected_indices.unsqueeze(-1).expand(-1, -1, self.gt_dict['img_overlap'].shape[-1]))
                coords = torch.gather(self.in_dict['coords_overlap'], dim=1,
                                      index=selected_indices.unsqueeze(-1).expand(-1, -1,
                                                                                  self.in_dict['coords_overlap'].shape[-1]))

            return {'idx': idx, 'all_coords': self.in_dict['coords'], 'coords': coords, 'indices': selected_indices}, \
                   {'img': img, 'raw_img': self.gt_dict['raw_img']}

        return self.in_dict, self.gt_dict


def build_video_edges_weight_tensor(shape, downsample, weight, frame_num=1, width_num=5, height_num=5):
    mask = torch.ones(shape)
    frame_indices = list(range(0, shape[0], downsample[0]))[1:]
    width_indices = list(range(0, shape[1], downsample[1]))[1:]
    height_indices = list(range(0, shape[2], downsample[2]))[1:]

    for index in frame_indices:
        mask[index - frame_num:index + frame_num, :, :] = weight
    mask[0, :, :] = weight
    mask[-1, :, :] = weight

    for index in width_indices:
        mask[:, index - width_num:index + width_num, :] = weight
    mask[:, 0, :] = weight
    mask[:, -1, :] = weight

    for index in height_indices:
        mask[:, :, index - height_num:index + height_num] = weight
    mask[:, :, 0] = weight
    mask[:, :, -1] = weight

    return mask


class Implicit3DWrapper(torch.utils.data.Dataset):
    def __init__(self, dataset, sidelength=None, sample_fraction=1., downsample=None, overlaps=None, weight_edges=None):
        if len(dataset) != 1:
            raise ValueError("Wrapper only supports a single 3D object")

        if isinstance(sidelength, int):
            sidelength = 3 * (sidelength,)

        self.dataset = dataset
        self.mgrid = get_mgrid(sidelength, dim=3)
        data = (torch.from_numpy(self.dataset[0]) - 0.5) / 0.5
        self.data = data.view(-1, self.dataset.channels)
        self.sample_fraction = sample_fraction
        self.num_groups = np.prod(downsample) if downsample is not None else 1
        self.N_samples = int(self.sample_fraction * self.mgrid.shape[0] / self.num_groups)
        self.loss_mask = None
        print("Using {} samples out of {}".format(self.N_samples, self.mgrid.shape[0]))

        self.in_dict, self.gt_dict = {'idx': 0, 'coords': self.mgrid, 'coords_overlap': self.mgrid}, \
                                     {'img': self.data, 'img_overlap': self.data}

        if downsample is None:
            self.coords_slices, self.gt_slices = None, None
        else:
            if weight_edges is not None:
                loss_mask = build_video_edges_weight_tensor(self.dataset.vid.shape, downsample, weight_edges)
                self.loss_mask, _ = partition_signal(loss_mask.reshape(self.data.shape), self.dataset.shape, downsample,
                                                     overlaps=overlaps)

            self.in_dict['coords'], self.coords_slices = partition_signal(self.mgrid,
                                                                          self.dataset.shape, downsample)
            self.gt_dict['img'], self.gt_slices = partition_signal(self.data,
                                                                   self.dataset.shape, downsample)

            self.in_dict['coords_overlap'], _ = partition_signal(self.mgrid, self.dataset.shape, downsample,
                                                                 overlaps=overlaps)
            self.gt_dict['img_overlap'], _ = partition_signal(self.data, self.dataset.shape, downsample,
                                                              overlaps=overlaps)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        mask = None
        if self.sample_fraction < 1.:
            if self.num_groups == 1:
                coord_idx = torch.randint(0, self.gt_dict['img'].shape[0], (self.N_samples,))
                data = self.gt_dict['img'][coord_idx, :]
                coords = self.in_dict['coords'][coord_idx, :]
            else:
                coord_idx = torch.randint(0, self.gt_dict['img_overlap'].shape[1], (self.num_groups, self.N_samples))
                data = torch.gather(self.gt_dict['img_overlap'], dim=1,
                                    index=coord_idx.unsqueeze(-1).expand(-1, -1, self.gt_dict['img_overlap'].shape[-1]))
                coords = torch.gather(self.in_dict['coords_overlap'], dim=1,
                                      index=coord_idx.unsqueeze(-1).expand(-1, -1,
                                                                           self.in_dict['coords_overlap'].shape[-1]))
                if self.loss_mask is not None:
                    mask = torch.gather(self.loss_mask, dim=1,
                                        index=coord_idx.unsqueeze(-1).expand(-1, -1, self.loss_mask.shape[-1]))
        else:
            data = self.gt_dict['img']
            coords = self.in_dict['coords']

        in_dict = {'idx': idx, 'coords': coords}
        if mask is None:
            gt_dict = {'img': data}
        else:
            gt_dict = {'img': data, 'mask': mask}

        return in_dict, gt_dict


def partition_signal(input, resolution, downsample, overlaps=None, selected_indices=None, pad_images=True):
    all_slices = []
    resolution = resolution + (input.shape[1],)
    downsample = downsample + (1,)

    if overlaps is None:
        overlaps = (0,) * len(downsample)
    else:
        overlaps = overlaps + (0,)

    if pad_images and len(resolution) == 3:
        padding = [0] * len(downsample)
        for i, (factor, size) in enumerate(zip(downsample, resolution)):
            if size % factor != 0:
                padding[i] = factor - (size % factor)
                print(f"Warning: Partition factor {factor} is not divided exactly by image dimension {size}, thus we pad with 'edge'")

        input = input.reshape(resolution)
        resolution = tuple([size + pad for size, pad in zip(resolution, padding)])
        input = torchvision.transforms.Pad((0, 0) + tuple(padding)[:-1], # left, top, right and bottom borders respectively.
                                           padding_mode='edge')(input.permute(2, 0, 1)).permute(1, 2, 0)
        input = input.reshape(-1, input.shape[-1])

    for factor, size, overlap in zip(downsample, resolution, overlaps):
        slices = []
        for i in range(factor):
            slice_start = i * (size // factor) - overlap
            slice_end = (i + 1) * (size // factor) + overlap
            if slice_start < 0:
                slice_end = slice_end - slice_start
                slice_start = 0
            if slice_end > size:
                slice_start = slice_start - (slice_end - size)
                slice_end = size

            slices.append(slice(slice_start, slice_end))
        all_slices.append(slices)

    all_slices = list(product(*all_slices))

    all_blocks = [input.reshape(resolution)[block_slice].reshape(-1, input.shape[1]) for block_slice in
                  all_slices]
    all_blocks = all_blocks if selected_indices is None else [all_blocks[i] for i in selected_indices]

    return torch.cat([b.unsqueeze(0) for b in all_blocks], dim=0), all_slices
