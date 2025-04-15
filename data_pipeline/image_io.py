import os
from typing import Union

import numpy as np
import pydicom
import SimpleITK as sitk
from data_pipeline.mri_sequence_handler import split_dwi


def read_image(img_path):
    """
    加载图像，支持DICOM和nii两种格式

    :param img_path: DICOM文件所在文件夹路径；nii文件路径；dicom文件路径列表；
    :return: SimpleITk Image
    """
    if isinstance(img_path, Union[list, tuple]):  # 输入路径列表
        slices = [pydicom.read_file(s, stop_before_pixels=True, force=True) for s in img_path]
        sorted_img_paths = [path for _, path in
                            sorted(zip(slices, img_path), key=lambda x: float(x[0].ImagePositionPatient[2]))]
        reader = sitk.ImageSeriesReader()
        reader.MetaDataDictionaryArrayUpdateOn()
        reader.LoadPrivateTagsOn()
        reader.SetFileNames(sorted_img_paths)
        img = reader.Execute()
    elif isinstance(img_path, str):
        if os.path.isdir(img_path):  # 输入文件夹地址
            reader = sitk.ImageSeriesReader()
            reader.MetaDataDictionaryArrayUpdateOn()
            reader.LoadPrivateTagsOn()
            series_IDs = sitk.ImageSeriesReader.GetGDCMSeriesIDs(img_path)
            dcm_names = reader.GetGDCMSeriesFileNames(img_path, series_IDs[0])
            reader.SetFileNames(dcm_names)
            img = reader.Execute()
        elif os.path.isfile(img_path) and img_path.endswith('.nii.gz'):  # 输入nii文件路径
            img = sitk.ReadImage(img_path)
        else:
            raise Exception('Unknown image format, read image failed. file path is {}'.format(img_path))
    else:
        raise Exception('Unknown data format, read image failed. file path is {}'.format(img_path))

    return img


def load_data(image_paths: dict,
              results: dict,
              ref_modalities: list):
    """
    读取多个序列的图像并放在字典中.
    """
    # 检查是否有输入没有拆分的DWI序列，如果有，进行拆分
    if 'DWI' in image_paths:
        dwi_paths = split_dwi(image_paths['DWI'], min_dwi_slices=10)
        del image_paths['DWI']
        image_paths.update(dwi_paths)

    # 读取数据
    result = {}
    for modality, image_path in image_paths.items():
        image_sitk = read_image(image_path)
        result[modality] = image_sitk

    # 读取数据信息，获取参考模态信息
    for modality in ref_modalities:
        if modality in result:
            ref_modality = modality
            break
    else:
        raise ValueError(f"参考模态数据不存在： {ref_modalities}.")
    image_sitk = result[ref_modality]
    shape = image_sitk.GetSize()[::-1]
    zero_arr = np.zeros(shape, dtype=np.int16)
    zero_image_sitk = sitk.GetImageFromArray(zero_arr)
    zero_image_sitk.CopyInformation(image_sitk)
    result['default_zero_image'] = zero_image_sitk
    results['raw_shape'] = shape
    results['spacing'] = image_sitk.GetSpacing()[::-1]
    results['raw_images_sitk'] = result


def write_sitk(image_arr, copy_image_itk=None, save_path=None):
    image_itk = sitk.GetImageFromArray(image_arr)
    if copy_image_itk:
        image_itk.CopyInformation(copy_image_itk)
    sitk.WriteImage(image_itk, save_path)


def save_reaults(save_dir, image_sitk, result_dict, save_name=None):
    os.makedirs(save_dir, exist_ok=True)
    prefix = save_name + '_' if save_name else ''
    save_path = os.path.join(save_dir, prefix + 'prostateMask.nii.gz')
    write_sitk(result_dict['coarse_prostate_mask_arr'], image_sitk, save_path)
    result_dict['ProstateMaskPath'] = save_path
    if 'lesions_mask_arr' in result_dict:
        save_path = os.path.join(save_dir, prefix + 'lesionMask.nii.gz')
        write_sitk(result_dict['lesions_mask_arr'], image_sitk, save_path)
        result_dict['LesionMaskPath'] = save_path


def save_raw_image(image_dict, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    for name, image in image_dict.items():
        save_path = os.path.join(save_dir, f'{name}.nii.gz')
        sitk.WriteImage(image, save_path)
