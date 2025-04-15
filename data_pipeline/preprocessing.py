import os

import numpy as np
import SimpleITK as sitk

import config


def crop_img(image_paths, mask_prostate_arr, crop_key, save_dir=None):
    """
    根据前列腺分割mask计算裁剪的坐标框，裁剪原始图像
    return：
    crop_images_sitk: 裁剪后的图像
    prostate_bbox: 裁剪使用的包围框坐标，zmin, ymin, xmin, zmax, ymax, xmax
    """
    def _crop_img(img_sitk, mask_shape, prostate_bbox, save_path=None):
        img_arr = sitk.GetArrayFromImage(img_sitk)
        assert mask_shape == img_arr.shape
        zmin, ymin, xmin, zmax, ymax, xmax = prostate_bbox
        img_crop_arr = img_arr[zmin: zmax + 1, ymin: ymax + 1, xmin: xmax + 1]
        img_crop_sitk = sitk.GetImageFromArray(img_crop_arr)
        img_crop_sitk.SetSpacing(img_sitk.GetSpacing())
        img_crop_sitk.SetDirection(img_sitk.GetDirection())
        if save_path:
            sitk.WriteImage(img_crop_sitk, save_path)
        return img_crop_sitk

    crop_images_sitk = {}
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    # 外扩固定尺寸，并进行边界情况处理
    d, h, w = mask_prostate_arr.shape
    spacing = list(image_paths.values())[0].GetSpacing()
    offset = int(config.CROP_PADDING[crop_key])
    offset_z = int(config.CROP_PADDING_mm / spacing[2])
    offset_y = int(config.CROP_PADDING_mm / spacing[1])
    offset_x = int(config.CROP_PADDING_mm / spacing[0])
    prostate_bbox = get_bounding_box(mask_prostate_arr)  # zmin, ymin, xmin, zmax, ymax, xmax
    prostate_bbox = prostate_bbox + [-offset, -offset, -offset, offset, offset, offset]
    # prostate_bbox = prostate_bbox + [-offset_z, -offset_y, -offset_x, offset_z, offset_y, offset_x]
    prostate_bbox = np.minimum(np.maximum(prostate_bbox, 0), [d - 1, h - 1, w - 1, d - 1, h - 1, w - 1])
    for modality, image_sitk in image_paths.items():
        save_path = os.path.join(save_dir, modality + '.nii.gz') if save_dir else None
        crop_images_sitk[modality] = _crop_img(image_sitk, mask_prostate_arr.shape, prostate_bbox, save_path=save_path)
    return crop_images_sitk, prostate_bbox


def get_bounding_box(mask):
    yx = np.any(mask, axis=0)
    zx = np.any(mask, axis=1)
    zmin, zmax = np.where(zx)[0][[0, -1]]
    ymin, ymax = np.where(yx)[0][[0, -1]]
    xmin, xmax = np.min(np.where(yx)[1]), np.max(np.where(yx)[1])
    return np.array([zmin, ymin, xmin, zmax, ymax, xmax])

def register(image_dict, target_image):
    registered_image_dict = {}
    for name, image in image_dict.items():
        registered_image = register_image_itk(image, target_image)
        registered_image_dict[name] = registered_image
    return registered_image_dict


def register_image_itk(ori_img, target_img, mode_name='img'):
    """
    用itk方法将原始图像resample到与目标图像一致
    :param ori_img: 原始需要对齐的itk图像
    :param target_img: 要对齐的目标itk图像
    :param resamplemethod: itk插值方法: sitk.sitkLinear-线性  sitk.sitkNearestNeighbor-最近邻
    :return:img_res_itk: 重采样好的itk图像
    """
    if mode_name == 'img':
        resamplemethod = sitk.sitkBSpline
    else:
        resamplemethod = sitk.sitkNearestNeighbor

    target_Size = target_img.GetSize()      # 目标图像大小  [x,y,z]
    target_Spacing = target_img.GetSpacing()   # 目标的体素块尺寸    [x,y,z]
    target_origin = target_img.GetOrigin()      # 目标的起点 [x,y,z]
    target_direction = target_img.GetDirection()  # 目标的方向 [冠,矢,横]=[z,y,x]
    # if target_origin == ori_img.GetOrigin() and target_Spacing == ori_img.GetSpacing() and target_Size == ori_img.GetSize():
    #     print('same info')

    # itk的方法进行resample
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(ori_img)  # 需要重新采样的目标图像
    # 设置目标图像的信息
    resampler.SetSize(target_Size)		# 目标图像大小
    resampler.SetOutputOrigin(target_origin)
    resampler.SetOutputDirection(target_direction)
    resampler.SetOutputSpacing(target_Spacing)
    # 根据需要重采样图像的情况设置不同的dype
    if resamplemethod == sitk.sitkNearestNeighbor:
        resampler.SetOutputPixelType(sitk.sitkUInt16)   # 近邻插值用于mask的，保存uint16
    else:
        resampler.SetOutputPixelType(sitk.sitkInt16)  # 线性插值用于PET/CT/MRI之类的，保存float32
    resampler.SetTransform(sitk.Transform(3, sitk.sitkIdentity))
    resampler.SetInterpolator(resamplemethod)
    itk_img_resampled = resampler.Execute(ori_img)  # 得到重新采样后的图像
    return itk_img_resampled
