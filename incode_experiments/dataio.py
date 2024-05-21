from itertools import product

import numpy as np
import scipy.ndimage
import scipy.special
import torchvision

np.float = np.float64
np.int = np.int_

import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose, ToTensor
from incode_experiments import utils
import matplotlib.pyplot as plt
import cv2


def get_mgrid(sidelen, dim=2, zero_to_one=False):
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

    if not zero_to_one:
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

class ImageFile(Dataset):
    def __init__(self, filename, fx=1, fy=1, size=None):
        super().__init__()
        if size is not None:
            self.img = utils.normalize(plt.imread(filename).astype(np.float32), True)
            self.img = cv2.resize(self.img, size, interpolation=cv2.INTER_AREA)
        else:
            self.img = utils.normalize(plt.imread(filename).astype(np.float32), True)
            self.img = cv2.resize(self.img, None, fx=fx, fy=fy, interpolation=cv2.INTER_AREA)
        self.img_channels = self.img.shape[-1] if len(self.img.shape) == 3 else 1

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return self.img


class ImageFileFromArray(Dataset):
    def __init__(self, img: np.array):
        super().__init__()
        self.img = img
        self.img_channels = self.img.shape[-1] if len(self.img.shape) == 3 else 1

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return self.img


class Implicit2DWrapper(torch.utils.data.Dataset):
    def __init__(self, dataset, compute_diff=None, zero_to_one=False):
        self.norm_transform = Compose([
            ToTensor(),
        ])

        self.compute_diff = compute_diff
        self.dataset = dataset
        self.sidelength = dataset[0].shape if len(dataset[0].shape) < 3 else dataset[0].shape[:2]
        self.mgrid = get_mgrid(self.dataset[0].shape, zero_to_one=zero_to_one)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        raw_image = self.dataset[idx]
        img = self.norm_transform(raw_image)

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
        img = self.norm_transform(raw_image)
        spatial_img = img.clone()
        img = img.permute(1, 2, 0).view(-1, self.dataset.img_channels)

        gt_dict = {'img': img, 'raw_img': np.array(raw_image)}

        return spatial_img, img, gt_dict


class ImplicitSingle2DWrapper(torch.utils.data.Dataset):
    def __init__(self, dataset, n_samples=None, compute_diff=None, downsample=None, overlaps=None, zero_to_one=False):
        if len(dataset) != 1:
            raise ValueError("Wrapper only supports a single 2D object")

        self.dataset = dataset
        self.dataset_wrapper = Implicit2DWrapper(dataset, compute_diff, zero_to_one=zero_to_one)
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
