import numpy as np
from scipy import ndimage
from skimage import measure


def keep_max_center_region(mask_arr):
    """
        过滤前列腺预测mask，仅保留靠近中心的一个连通域
    """
    min_volume = 1000
    bi_mask = mask_arr > 0
    labeled = measure.label(bi_mask, connectivity=2)  # 8连通
    props = measure.regionprops(labeled)
    image_center = [d / 2 for d in mask_arr.shape]
    region_centers = [prop.centroid for prop in props]
    distances = [np.linalg.norm(np.array(region_center) - np.array(image_center)) for region_center in region_centers]
    distances = [distance if prop.area > min_volume else 10000 for distance, prop in zip(distances, props)]
    index = np.argmin(distances)
    bi_mask[labeled != index + 1] = 0
    return bi_mask


def coarse_seg_post_process(mask_arr):
    bi_mask = keep_max_center_region(mask_arr)
    return bi_mask.astype(np.uint8)


def get_prostate_max_layer(mask_arr):
    areas = []
    for layer in range(len(mask_arr)):
        area = np.sum(mask_arr[layer] > 0)
        areas.append(area)
    max_layer = np.argmax(areas)
    return max_layer


def lesion_seg_post_process(mask_arr, image_dict, result_dict):
    """
    病灶分割后处理
    """
    # 1. 将基于裁剪图像的病灶分割结果映射回原始图像
    raw_size_mask_arr = np.zeros(image_dict['raw_shape'])
    crop_bbox = image_dict['crop_bbox_for_lesion']
    zmin, ymin, xmin, zmax, ymax, xmax= crop_bbox
    raw_size_mask_arr[zmin: zmax + 1, ymin: ymax + 1, xmin: xmax + 1] = mask_arr

    # 2. 过滤小的连通域，过滤腺体外连通域，并对病灶进行编号
    min_volume_threshold = 20
    large_volume_threshold = 20000
    labeled, n = measure.label(raw_size_mask_arr, connectivity=2, return_num=True)
    spacing = image_dict['spacing']
    props = measure.regionprops(labeled)
    areas = [prop.area for prop in props]
    filter_mask_arr = np.zeros_like(raw_size_mask_arr, dtype=np.uint8)
    kernel = np.ones((1, 3, 3), dtype=np.uint8)  # TODO：合适的参数确定（1，3，3）？（3，5，5）
    prostate_arr = ndimage.binary_dilation(result_dict['coarse_prostate_mask_arr'],
                                           kernel, iterations=1).astype(np.uint8)
    number = 1  # 病灶编号
    for i in range(n):
        volume = areas[i] * np.prod(spacing)
        volume_in_prostate = np.sum(prostate_arr[labeled == i + 1])
        # volume >= min_volume_threshold 超过病灶最小体积的可以保留
        # volume_in_prostate > areas[i] * 0.3 病灶30%以上在前列腺腺体内，整体保留
        # volume > large_volume_threshold 较大的病灶直接保留，这种情况腺体分割可能会不准确，按照腺体交集判断会被滤掉
        if (volume >= min_volume_threshold and volume_in_prostate / areas[i] > 0.3) or volume > large_volume_threshold:
            filter_mask_arr[labeled == i + 1] = number
            number += 1

    # 计算每个病灶的最大层面所在的层
    lesion_max_layers = []
    lesion_ranges = []
    for lesion_id in range(1, number):
        areas = []
        for layer in range(len(filter_mask_arr)):
            area = np.sum(filter_mask_arr[layer] == lesion_id)
            areas.append(area)
        max_layer = np.argmax(areas)
        lesion_max_layers.append(max_layer)

        zx = np.any(filter_mask_arr == lesion_id, axis=1)
        zmin, zmax = np.where(zx)[0][[0, -1]]
        lesion_ranges.append([zmin, zmax])
    result_dict['lesion_max_layers'] = lesion_max_layers
    result_dict['lesion_ranges'] = lesion_ranges
    return filter_mask_arr