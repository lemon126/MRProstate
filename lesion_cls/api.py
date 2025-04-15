import copy
from collections import OrderedDict
import importlib

import numpy as np
from scipy.ndimage import zoom
import torch

from . import hardnet
from . import configs


class Classifier(object):
    def __init__(self, model_path_dict, device):
        self.config = configs.configs()
        self.model = hardnet.Net(self.config, logger=None)
        self.model.eval()
        self.model.to(device)
        load_checkpoint(self.model, model_path_dict['model_ckpt_path'], mode='test')
        self.device = device
        self.modalities = ['T2', 'T2fs', 'ADC', 'DWI1000', 'DWI1500', 'DWI2000', 'DWI3000', 'lesionMask']
        self.input_mask = True
        self.input_size = self.config.input_size
        self.thr = 0.4788

    def run_inference(self, image_arr, mask_arr, spacing):
        n_lesions = int(np.max(mask_arr))
        if self.input_mask:
            images = np.concatenate((image_arr, np.expand_dims(mask_arr, axis=0)), axis=0)
        else:
            images = copy.deepcopy(image_arr)
        images = resize_and_normalize(images, self.modalities, spacing, self.config.target_spacing_zyx)
        mask_arr_zoom = copy.deepcopy(images[-1])  # TODO: self.input_mask判断
        probs, classes = [], []
        for i in range(1, n_lesions + 1):
            if self.input_mask:
                images[-1] = copy.deepcopy((mask_arr_zoom == i).astype(np.int8))
            zmin, zmax, ymin, ymax, xmin, xmax = get_crop_border(images[-1])
            roi = images[:, zmin: zmax, ymin: ymax, xmin: xmax]
            input_images = torch.tensor(roi).unsqueeze(0).float().to(torch.device(self.device))
            prob = self.model({'inputs': input_images}, phase='test')['predict_label'][0][1].detach().cpu().numpy()
            probs.append(prob)
            classes.append(1 if prob > self.thr else 0)
        return probs, classes


def resize_and_normalize(image_arr, modalities, spacing, target_spacing):
    new_image_arr = []
    scale = np.array(spacing) / np.array(target_spacing)
    for i, (data, modality) in enumerate(zip(image_arr, modalities)):
        order = 1 if 'mask' not in modality.lower() else 0
        data = zoom(data, scale, order=order)
        if 'mask' not in modality.lower():
            mean_ = np.mean(data)
            std_ = np.std(data)
            data = (data - mean_) / (std_ + 1e-5)
        new_image_arr.append(data)
    return np.stack(new_image_arr, axis=0)


def get_bounding_box(mask):
    yx = np.any(mask, axis=0)
    zx = np.any(mask, axis=1)
    zmin, zmax = np.where(zx)[0][[0, -1]]
    ymin, ymax = np.where(yx)[0][[0, -1]]
    xmin, xmax = np.min(np.where(yx)[1]), np.max(np.where(yx)[1])
    return np.array([zmin, ymin, xmin, zmax + 1, ymax + 1, xmax + 1])


def get_crop_border(mask):
    pad_xy = 5
    pad_z = 2
    min_length_xy = 128  # 64
    min_length_z = 16  # 8
    box = get_bounding_box(mask)
    box = np.array(box) + np.array([-pad_z, -pad_xy, -pad_xy, pad_z, pad_xy, pad_xy])
    zmin, ymin, xmin, zmax, ymax, xmax = box
    if ymax - ymin < min_length_xy:
        ymin -= (min_length_xy - ymax + ymin) // 2
        ymax = ymin + min_length_xy
    if xmax - xmin < min_length_xy:
        xmin -= (min_length_xy - xmax + xmin) // 2
        xmax = xmin + min_length_xy
    if zmax - zmin < min_length_z:
        zmin -= (min_length_z - zmax + zmin) // 2
        zmax = zmin + min_length_z

    xmin = max(xmin, 0)
    ymin = max(ymin, 0)
    zmin = max(zmin, 0)

    return zmin, zmax, ymin, ymax, xmin, xmax


def import_module(name, path):
    """
    correct way of importing a module dynamically in python 3.
    :param name: name given to module instance.
    :param path: path to module.
    :return: module: returned module instance.
    """
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def load_checkpoint(net, checkpoint_path, mode='resume', optimizer=None, cf=None, ):
    checkpoint = torch.load(checkpoint_path)
    if 'state_dict' in checkpoint:
        state_dict = get_clean_state_dict(checkpoint['state_dict'])
    else:
        state_dict = get_clean_state_dict(checkpoint)

    if mode == 'test':
        net.load_state_dict(state_dict)
        net.eval()
    elif mode == 'pretrain':
        net_state_dict = net.state_dict()
        pretrain_dict = {k: v for k, v in state_dict.items() if
                         (k in net_state_dict.keys() and k not in cf.exclude_pretrain_names)}
        net_state_dict.update(pretrain_dict)
        net.load_state_dict(net_state_dict)
    elif mode == 'resume':
        net.load_state_dict(state_dict)
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer'])
        return checkpoint['epoch'], checkpoint['metrics']
    else:
        raise ValueError('load checkpoint mode must be one of test, pretrain or resume!')

def get_clean_state_dict(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k
        if k[:7] == 'module.':
            name = k[7:]  # remove `module.`
        new_state_dict[name] = v
    return new_state_dict
