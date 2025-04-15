import os
from typing import Union

import numpy as np
import SimpleITK as sitk


def find_dwi(modality, data_dict, max_b_diff=300):
    target_b_value = int(modality.replace('DWI', ''))
    b_values = np.array(
        [int(mod.replace('DWI', '')) for mod in data_dict.keys() if 'DWI' in mod and 'DWI' != mod])
    if len(b_values) > 0:
        b_values = np.sort(b_values)
        match_value = b_values[np.argmin(np.abs(b_values - target_b_value))]
        if abs(match_value - target_b_value) <= max_b_diff:
            return 'DWI{}'.format(match_value)
    return None


def select_modalities(data_dict, target_modalities):
    """
    根据target_modalities挑选出需要的image并组成列表，不存在的模态用默认0值image填充。
    """
    matched_names = []
    images = []
    for modality in target_modalities:
        match_name = 'default_zero_image'
        if isinstance(modality, str):
            modality = [modality]
        for name in modality:
            if name in data_dict:
                match_name = name
                break
        else:
            if modality[0] == 'DWI':  # 如果模态等于DWI，默认返回高B值
                match_dwi = find_dwi('DWI3000', data_dict, max_b_diff=3000)
                if match_dwi:
                    match_name = match_dwi
            elif modality[0].startswith('DWI'):
                match_dwi = find_dwi(modality[0], data_dict)
                if match_dwi:
                    match_name = match_dwi
        images.append(data_dict[match_name])
        matched_names.append(match_name)
    print('原始通道：', data_dict.keys())
    print('目标通道：', target_modalities)
    print('实际采用通道：', matched_names)
    return images


B_VALUE_KEYS = {
    'GE': '0043|1039',
    'SIEMENS': '0019|100c',
    'PHILIPS': '0018|9087',
    'UIH': '0018|9087',
}


def is_subsequence(small, large):
    it = iter(large)
    return all(char in it for char in small)


def find_closest_key(input_string, keys):
    input_string_upper = input_string.upper()
    for key in keys:
        if is_subsequence(key, input_string_upper):
            return key
    return None


def get_b_value(dcm_path):
    # 获取dcm文件的设备厂商
    reader = sitk.ImageFileReader()
    reader.SetFileName(dcm_path)
    reader.LoadPrivateTagsOn()
    reader.ReadImageInformation()
    manufacturer = reader.GetMetaData('0008|0070').strip().upper()

    # 适配查找该厂商的B值保存字段
    if manufacturer in B_VALUE_KEYS:
        b_value_key = B_VALUE_KEYS[manufacturer]
    else:
        manufacturer = find_closest_key(manufacturer, B_VALUE_KEYS.keys())
        if not manufacturer:
            raise RuntimeError(f"'{manufacturer}' does not match with any of B_VALUE_KEYS: {list(B_VALUE_KEYS.keys())}")
        b_value_key = B_VALUE_KEYS[manufacturer]

    # 从相应的B值字段提取B值信息，并进行后处理
    base_b_value = reader.GetMetaData(b_value_key).strip()
    raw_b_value = base_b_value.split('\\')[0]
    if 'GE' in manufacturer and int(raw_b_value) > 5000:  # 处理‘1000000500’的情况
        # print('raw_b_value：', raw_b_value, '识别B值为：', int(raw_b_value[1:]))
        b_value = str(int(raw_b_value[1:]))
    else:
        b_value = str(raw_b_value)

    return b_value


def split_dwi(dicom_dir, min_dwi_slices=10):
    """获取输入目录下的文件，根据B值拆分DWI序列"""
    dcm_list = os.listdir(dicom_dir)
    if 'DIRFILE' in dcm_list:
        dcm_list.remove('DIRFILE')
    dwi_dict = {}
    for dcm_name in dcm_list:
        dcm_path = os.path.join(dicom_dir, dcm_name)
        try:
            b_value = get_b_value(dcm_path)
        except Exception as e:
            print(f"An error occurred while getting B value from '{dcm_path}': {e}, skip!")
            continue
        b_key = 'DWI' + b_value
        if b_key not in dwi_dict:
            dwi_dict[b_key] = [dcm_path]
        else:
            dwi_dict[b_key].append(dcm_path)
    # 检查最少文件数目是否符合要求
    dwi_dict = {name: paths for name, paths in dwi_dict.items() if len(paths) >= min_dwi_slices}
    return dwi_dict


def is_exist_modality(src: str, target: Union[tuple, list]):
    if 'DWI' == src:
        match = [item for item in target if src in item]
    else:
        match = [item for item in target if src == item]
    if match:
        return True
    return False


def check_required_modalities(required_fields, data):
    # 检查必须字段是否具备
    all_fields = list(data.keys())
    for required_field in required_fields:
        if isinstance(required_field, str) and not is_exist_modality(required_field, all_fields):
            return required_field
        if isinstance(required_field, Union[tuple, list]): # 如果是tuple或list，那么其中的内容有一个存在即满足要求
            for name in required_field:
                if is_exist_modality(name, all_fields):
                    break
            else:
                return required_field
