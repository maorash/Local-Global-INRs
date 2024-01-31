import os

import cmapy
import cv2
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import scipy.io.wavfile as wavfile
import torch
from torch.autograd import grad
from torchvision.utils import make_grid

import dataio


def laplace(y, x):
    grad = gradient(y, x)
    return divergence(grad, x)


def divergence(y, x):
    div = 0.
    for i in range(y.shape[-1]):
        div += grad(y[..., i], x, torch.ones_like(y[..., i]), create_graph=True)[0][..., i:i+1]
    return div


def gradient(y, x, grad_outputs=None):
    if grad_outputs is None:
        grad_outputs = torch.ones_like(y)
    grad = torch.autograd.grad(y, [x], grad_outputs=grad_outputs, create_graph=True)[0]
    return grad


def cond_mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def decode_video(model, coord_dataset, is_lc_lg):
    resolution = coord_dataset.dataset.vid.shape
    with torch.no_grad():
        if is_lc_lg:
            Nslice = 100
            output = torch.zeros(coord_dataset.gt_dict['img'].shape)
            coords = coord_dataset.in_dict['coords'].cuda()
            split = coords.shape[1] / Nslice
            extra_step = 0 if split.is_integer() else 1
            split = int(split)
            for i in range(Nslice + extra_step):
                pred = model({'coords': coords[:, i * split:(i + 1) * split, :].unsqueeze(0)})['model_out'].squeeze(0)
                output[:, i * split:(i + 1) * split, :] = pred.cpu()

            pred_vid = dataio.lin2vid(output, resolution, coord_dataset.gt_slices)
            pred_vid = pred_vid / 2 + 0.5
            pred_vid = torch.clamp(pred_vid, 0, 1)
            return pred_vid.detach().cpu().numpy()
        else:
            raise ValueError("Not implemented yet")


def write_video_summary(coord_dataset, is_lc_lg, model, model_input, gt, model_output, writer, total_steps,
                        prefix='train_', decode_all=True):
    resolution = coord_dataset.dataset.vid.shape
    frames = [10, 30, 90, 150, 165, 180, 200, 230, 250, 265, 285, 299]
    gt_vid = torch.from_numpy(coord_dataset.dataset.vid)

    with torch.no_grad():
        if decode_all:
            Nslice = 200
            if is_lc_lg:
                output = torch.zeros(coord_dataset.gt_dict['img'].shape)
                coords = coord_dataset.in_dict['coords'].cuda()
            else:
                output = torch.zeros(coord_dataset.gt_dict['img'].unsqueeze(0).shape)
                coords = coord_dataset.in_dict['coords'].unsqueeze(0).cuda()
            split = int(coords.shape[1] / Nslice)
            for i in range(Nslice):
                pred = model({'coords': coords[:, i * split:(i + 1) * split, :].unsqueeze(0)})['model_out'].squeeze(0)
                output[:, i * split:(i + 1) * split, :] = pred.cpu()

            pred_vid = dataio.lin2vid(output, resolution, coord_dataset.gt_slices)
        else:
            if is_lc_lg:
                raise ValueError("Not implemented yet")
            Nslice = 10
            coords = [dataio.get_mgrid((1, resolution[1], resolution[2]), dim=3)[None, ...].cuda() for f in frames]
            for idx, f in enumerate(frames):
                coords[idx][..., 0] = (f / (resolution[0] - 1) - 0.5) * 2
            coords = torch.cat(coords, dim=0)

            output = torch.zeros(coords.shape)
            split = int(coords.shape[1] / Nslice)
            for i in range(Nslice):
                pred = model({'coords': coords[:, i * split:(i + 1) * split, :]})['model_out']
                output[:, i * split:(i + 1) * split, :] = pred.cpu()
                pred_vid = output.view(len(frames), resolution[1], resolution[2], 3)

            gt_vid = gt_vid[frames]

    pred_vid = pred_vid / 2 + 0.5
    pred_vid = torch.clamp(pred_vid, 0, 1)

    pred_vid = pred_vid.permute(0, 3, 1, 2)
    gt_vid = gt_vid.permute(0, 3, 1, 2)

    psnr = 10 * torch.log10(1 / torch.mean((gt_vid - pred_vid) ** 2))

    min_max_summary(prefix + 'coords', model_input['coords'], writer, total_steps)
    min_max_summary(prefix + 'pred_vid', pred_vid, writer, total_steps)
    writer.add_scalar(prefix + "psnr", psnr, total_steps)

    # Now we want to log specific frames
    if decode_all:
        pred_vid = pred_vid[frames]
        gt_vid = gt_vid[frames]

    for i, frame in enumerate(frames):
        # Log all the frames individually
        writer.add_image(prefix + f'output_frame_{frame}', make_grid(pred_vid[i], scale_each=False, normalize=True),
                         global_step=total_steps)
        frame_psnr = 10 * torch.log10(1 / torch.mean((gt_vid[i] - pred_vid[i]) ** 2))
        writer.add_scalar(prefix + f'frame_{frame}_psnr', frame_psnr, total_steps)

    output_vs_gt = torch.cat((gt_vid, pred_vid), dim=-2)
    writer.add_image(prefix + 'output_vs_gt', make_grid(output_vs_gt, scale_each=False, normalize=True),
                     global_step=total_steps)


def write_image_summary(image_resolution, coords_slices, gt_slices, model, model_input, gt,
                        model_output, writer, total_steps, prefix='train_'):
    gt_img = dataio.lin2img(gt['img'], image_resolution, gt_slices)
    pred_img = dataio.lin2img(model_output['model_out'], image_resolution, gt_slices)

    img_gradient = gradient(model_output['model_out'], model_output['model_in'])
    img_laplace = laplace(model_output['model_out'], model_output['model_in'])

    output_vs_gt = torch.cat((gt_img, pred_img), dim=-1)
    writer.add_image(prefix + 'gt_vs_pred', make_grid(output_vs_gt, scale_each=False, normalize=True),
                     global_step=total_steps)

    pred_img = dataio.rescale_img((pred_img + 1) / 2, mode='clamp').permute(0, 2, 3, 1).squeeze(
        0).detach().cpu().numpy()
    pred_grad = dataio.grads2img(dataio.lin2img(img_gradient, slices=coords_slices)).permute(1, 2,
                                                                                             0).squeeze().detach().cpu().numpy()
    pred_lapl = cv2.cvtColor(cv2.applyColorMap(dataio.to_uint8(dataio.rescale_img(
        dataio.lin2img(img_laplace, slices=coords_slices), perc=2).permute(0, 2, 3, 1).squeeze(
        0).detach().cpu().numpy()), cmapy.cmap('RdBu')), cv2.COLOR_BGR2RGB)

    gt_img = dataio.rescale_img((gt_img + 1) / 2, mode='clamp').permute(0, 2, 3, 1).squeeze(0).detach().cpu().numpy()
    gt_grad = dataio.grads2img(dataio.lin2img(gt['gradients'])).permute(1, 2, 0).squeeze().detach().cpu().numpy()
    gt_lapl = cv2.cvtColor(cv2.applyColorMap(dataio.to_uint8(dataio.rescale_img(
        dataio.lin2img(gt['laplace']), perc=2).permute(0, 2, 3, 1).squeeze(0).detach().cpu().numpy()),
                                             cmapy.cmap('RdBu')), cv2.COLOR_BGR2RGB)

    writer.add_image(prefix + 'pred_img', torch.from_numpy(pred_img).permute(2, 0, 1), global_step=total_steps)
    writer.add_image(prefix + 'pred_grad', torch.from_numpy(pred_grad).permute(2, 0, 1), global_step=total_steps)
    writer.add_image(prefix + 'pred_lapl', torch.from_numpy(pred_lapl).permute(2, 0, 1), global_step=total_steps)
    writer.add_image(prefix + 'gt_img', torch.from_numpy(gt_img).permute(2, 0, 1), global_step=total_steps)
    writer.add_image(prefix + 'gt_grad', torch.from_numpy(gt_grad).permute(2, 0, 1), global_step=total_steps)
    writer.add_image(prefix + 'gt_lapl', torch.from_numpy(gt_lapl).permute(2, 0, 1), global_step=total_steps)

    write_image_psnr(dataio.lin2img(model_output['model_out'], image_resolution, gt_slices),
                     gt['raw_img'], writer, total_steps, prefix + 'img_')


def write_audio_summary(logging_root_path, model, model_input, gt, model_output, writer, total_steps, prefix='train'):
    gt_func = torch.squeeze(gt['func'].reshape(1, -1, 1))
    gt_rate = torch.squeeze(gt['rate']).detach().cpu().numpy()
    gt_scale = torch.squeeze(gt['scale']).detach().cpu().numpy()
    pred_func = torch.squeeze(model_output['model_out'].reshape(1, -1, 1))
    coords = torch.squeeze(model_output['model_in'].reshape(1, -1, 1).clone()).detach().cpu().numpy()

    fig, axes = plt.subplots(3, 1)

    strt_plot, fin_plot = int(0.05 * len(coords)), int(0.95 * len(coords))
    coords = coords[strt_plot:fin_plot]
    gt_func_plot = gt_func.detach().cpu().numpy()[strt_plot:fin_plot]
    pred_func_plot = pred_func.detach().cpu().numpy()[strt_plot:fin_plot]

    axes[1].plot(coords, pred_func_plot)
    axes[0].plot(coords, gt_func_plot)
    axes[2].plot(coords, gt_func_plot - pred_func_plot)

    axes[0].get_xaxis().set_visible(False)
    axes[1].axes.get_xaxis().set_visible(False)
    axes[2].axes.get_xaxis().set_visible(False)

    writer.add_figure(prefix + 'gt_vs_pred', fig, global_step=total_steps)

    min_max_summary(prefix + 'coords', model_input['coords'], writer, total_steps)
    min_max_summary(prefix + 'pred_func', pred_func, writer, total_steps)
    min_max_summary(prefix + 'gt_func', gt_func, writer, total_steps)

    # write audio files:
    wavfile.write(os.path.join(logging_root_path, 'gt.wav'), gt_rate, gt_func.detach().cpu().numpy())
    wavfile.write(os.path.join(logging_root_path, 'pred.wav'), gt_rate, pred_func.detach().cpu().numpy())

    def _calc_audio_psnr(original, compressed):
        mse = np.mean((original - compressed) ** 2)
        if mse == 0:
            return 100
        psnr = 20 * np.log10(1 / np.sqrt(mse))
        return psnr

    writer.add_scalar(prefix + "_psnr", _calc_audio_psnr(pred_func.detach().cpu().numpy(), gt_func.detach().cpu().numpy()), total_steps)


def min_max_summary(name, tensor, writer, total_steps):
    writer.add_scalar(name + '_min', tensor.min().detach().cpu().numpy(), total_steps)
    writer.add_scalar(name + '_max', tensor.max().detach().cpu().numpy(), total_steps)


def write_image_psnr(pred_img, gt_img, writer, iter, prefix):
    batch_size = pred_img.shape[0]

    pred_img = pred_img.detach().cpu().numpy()
    gt_img = gt_img.detach().cpu().numpy()

    while len(gt_img.shape) < 4:
        gt_img = np.expand_dims(gt_img, axis=0)

    psnrs, ssims = list(), list()
    for i in range(batch_size):
        p = pred_img[i].transpose(1, 2, 0)
        trgt = gt_img[i].transpose(1, 2, 0)

        psnr = calc_psnr_image(p, trgt)
        psnrs.append(psnr)

    writer.add_scalar(prefix + "psnr", np.mean(psnrs), iter)


def calc_psnr_image(original, compressed):
    original = cv.normalize(original.astype('float32'), None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX,
                            dtype=cv2.CV_32F)
    compressed = cv.normalize(compressed.astype('float32'), None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX,
                              dtype=cv2.CV_32F)
    mse = np.mean((original - compressed) ** 2)
    if mse == 0:
        return 100
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr
