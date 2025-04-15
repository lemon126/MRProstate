import os
import logging
import logging.config
from typing import Union

import numpy as np
import SimpleITK as sitk
import torch
import yaml

import config
from config import MODEL_PATHS
from config import CONFIG_PATHS
from lesion_seg import Segmentor
from lesion_cls.api import Classifier
from data_pipeline.image_io import load_data
from data_pipeline.image_io import save_reaults
from data_pipeline.preprocessing import register
from data_pipeline.mri_sequence_handler import select_modalities
from data_pipeline.mri_sequence_handler import check_required_modalities
from data_pipeline.preprocessing import crop_img
from data_pipeline.postprocessing import coarse_seg_post_process
from data_pipeline.postprocessing import lesion_seg_post_process


class MRProstate:
    def __init__(self, settings: dict[str, bool]=None):
        """
        初始化并加载模型(模型路径在配置文件中进行修改)。
        :param settings: 选择是否执行所有模块，格式如下：
                        {
                            "do_lesion_segmentation": True,
                            "do_lesion_bm_classification": True
                        }
                        默认所有的模块都会执行。
        """
        MRProstate.setup_logging_system(CONFIG_PATHS["log"])
        logging.info('Logging system initialization completed')

        self.result_dict, self.image_dict = {}, {}
        device = torch.device('cuda', 0)

        self.settings = settings if settings else {
            "do_lesion_segmentation": True,
            "do_lesion_bm_classification": True
        }

        logging.info('---> Enter MRProstate initialization')
        logging.info('Initializing prostate-coarse-segmentation')
        self.coarse_segmentor = Segmentor(MODEL_PATHS["CoarseSeg"], device)

        if self.settings['do_lesion_segmentation']:
            logging.info('Initializing prostate-lesion-segmentation')
            self.lesion_segmentor = Segmentor(MODEL_PATHS["LesionSeg"], device, True)

        if self.settings['do_lesion_bm_classification']:
            logging.info('Initializing prostate-lesion-bm-classification')
            self.lesion_bm_classifier = Classifier(MODEL_PATHS["LesionBMCls"],
                                                   device)

    def _run_coarse_segmentation(self, exist_prosatate_mask=None, save_path=None):
        if 'prostateMask' in self.image_dict['raw_images_sitk']:
            mask_sitk = self.image_dict['raw_images_sitk']['prostateMask']
            mask_arr = sitk.GetArrayFromImage(mask_sitk)
            logging.info('Since a prostate mask exists, coarse prostate segmentation will be skipped, '
                         'and the existing mask will be used instead.')
        elif exist_prosatate_mask and os.path.isfile(exist_prosatate_mask):
            mask_sitk = sitk.ReadImage(exist_prosatate_mask)
            mask_arr = sitk.GetArrayFromImage(mask_sitk)
            logging.info('Since a prostate mask exists, coarse prostate segmentation will be skipped, '
                         'and the existing mask will be used instead.')
        else:
            missing_field = check_required_modalities(config.PROSTATE_SEG_REQUIRED_FIELDS,
                                                      self.image_dict['raw_images_sitk'])
            if missing_field is not None:
                raise ValueError('Missing required field: {}'.format(missing_field))
            image_sitk = select_modalities(self.image_dict['raw_images_sitk'], config.PROSTATE_SEG_FIELDS)[0]
            mask_sitk = self.coarse_segmentor.run_inference(image_sitk, save_path=save_path)
            mask_arr = sitk.GetArrayFromImage(mask_sitk)
            mask_arr = coarse_seg_post_process(mask_arr)
        self.result_dict['coarse_prostate_mask_arr'] = mask_arr
        self.result_dict['prostate_volume(cm^3)'] = np.sum(mask_arr) * np.prod(self.image_dict['spacing']) / 1000
        return mask_arr

    def _crop_prostate(self):
        mask_arr = self.result_dict['coarse_prostate_mask_arr']

        crop_images_sitk, prostate_bbox = crop_img(self.image_dict['raw_images_sitk'],
                                                   mask_arr,
                                                   crop_key='for_lesion_seg',
                                                   save_dir=None)
        self.image_dict['images_for_lesion'] = crop_images_sitk
        self.image_dict['crop_bbox_for_lesion'] = prostate_bbox

    def _run_lesion_segmentation(self, exist_lesion_mask=None, save_path=None):
        if exist_lesion_mask and os.path.isfile(exist_lesion_mask):
            mask_sitk = sitk.ReadImage(exist_lesion_mask)
            mask_arr = sitk.GetArrayFromImage(mask_sitk)
            # mask_arr, n = measure.label(mask_arr, connectivity=2, return_num=True)  ## TODO: remove
            logging.info('Since a lesion mask exists, lesion segmentation will be skipped, '
                         'and the existing mask will be used instead.')
        else:
            missing_field = check_required_modalities(config.LESION_SEG_REQUIRED_FIELDS,
                                                      self.image_dict['images_for_lesion'])
            if missing_field is not None:
                logging.info(f'Missing required_modalities: {missing_field}, lesion segmentation will be skipped.')
                return
            images = select_modalities(self.image_dict['images_for_lesion'], config.LESION_SEG_FIELDS)
            image_arr = [sitk.GetArrayFromImage(image) for image in images]
            image_arr = np.array(image_arr)
            props = {
                'sitk_stuff': {
                    'spacing': images[0].GetSpacing(),
                    'origin': images[0].GetOrigin(),
                    'direction': images[0].GetDirection()
                },
                'spacing': images[0].GetSpacing()[::-1]
            }
            mask_sitk = self.lesion_segmentor.run_inference(image_arr, props, save_path=save_path)
            mask_arr = sitk.GetArrayFromImage(mask_sitk)
            mask_arr = lesion_seg_post_process(mask_arr, self.image_dict, self.result_dict)
        self.result_dict['lesions_mask_arr'] = mask_arr

        # calculate lesion volumes
        self.result_dict['number_of_lesions'] = int(np.max(mask_arr))
        pixel_volume = np.prod(self.image_dict['spacing']) / 1000
        self.result_dict['lesion_volumes(cm^3)'] = [np.sum(mask_arr == i + 1) * pixel_volume
                                                    for i in range(int(np.max(mask_arr)))]

    def _run_lesion_bm_cls(self):
        missing_field = check_required_modalities(config.LESION_BM_CLS_REQUIRED_FIELDS,
                                                  self.image_dict['images_for_lesion'])
        if missing_field is not None:
            logging.info(f'Missing required_modalities: {missing_field}, lesion BM classification will be skipped.')
            return
        images = select_modalities(self.image_dict['images_for_lesion'], config.LESION_BM_CLS_FIELDS)
        image_arr = [sitk.GetArrayFromImage(image) for image in images]
        image_arr = np.array(image_arr)
        zmin, ymin, xmin, zmax, ymax, xmax = self.image_dict['crop_bbox_for_lesion']
        mask_roi_arr = self.result_dict['lesions_mask_arr'][zmin: zmax + 1, ymin: ymax + 1, xmin: xmax + 1]
        scores, bm_cls = self.lesion_bm_classifier.run_inference(image_arr,
                                                                 mask_roi_arr,
                                                                 self.image_dict['spacing'])
        self.result_dict['lesions_bm_scores'] = scores
        self.result_dict['lesions_bm_cls'] = bm_cls
        self.result_dict['patient_bm_score'] = np.max(scores) if scores else None
        self.result_dict['patient_bm_cls'] = np.max(bm_cls) if scores else None

    def _save_result(self, save_dir, save_name):
        if 'T2' in self.image_dict['raw_images_sitk']:
            ref_image_sitk = self.image_dict['raw_images_sitk']['T2']
        elif 'T2fs' in self.image_dict['raw_images_sitk']:
            ref_image_sitk = self.image_dict['raw_images_sitk']['T2fs']
        else:
            ref_image_sitk = None
        save_reaults(save_dir, ref_image_sitk, self.result_dict, save_name)

    def _register(self):
        target_name = None
        for name in config.REGISTER_TARGET:
            if name in self.image_dict['raw_images_sitk']:
                target_image = self.image_dict['raw_images_sitk'][name]
                target_name = name
                break
        logging.info('Performing image registration of all sequences to the {} sequence.'.format(target_name))
        self.image_dict['raw_images_sitk'] = register(self.image_dict['raw_images_sitk'], target_image)

    def run(self,
            file_paths: dict[str, Union[str, list, tuple]],
            save_dir: str=None,
            save_name: str=None,
            exist_lesion_mask: str=None,
            exist_prosatate_mask: str=None,
            is_registered: bool=False,
    ):
        """
        前列腺腺体分割、分区，病灶分割、良恶性分类推理计算。
        :param file_paths: 输入nii或dicom文件夹路径
        :param save_dir: 结果文件保存路径
        :param save_name: 结果文件保存前缀ID
        :param exist_lesion_mask: 如果存在已经生成的病灶mask，则直接加载，不重新计算
        :param exist_prosatate_mask: 如果存在已经生成的腺体分割mask，则直接加载，不重新计算
        :param is_registered: file_paths中输入的多参数MRI序列是否已经经过配准，如果为True，则认为数据已经配准，不执行配准步骤，
                              否则执行配准步骤，将多序列配准至T2/T2fs序列。


        file_paths = {
            ‘T2’: '/data/T2.nii.gz',
            ‘ADC’: '/data/ADC.nii.gz',
            ‘DWI1000’: '/data/DWI1000.nii.gz',
            ‘DWI2000’: '/data/DWI2000.nii.gz',
        }
        return:
        {
            'coarse_prostate_mask_arr': np.ndarray,  # 前列腺腺体mask(0背景1腺体)
            'lesions_mask_arr': np.ndarray,  # 病灶分割mask(分别用1，2，3等表示不同的病灶，和病灶信息list顺序一致)
            'prostate_max_layer': int, # 腺体最大层面
            'lesion_max_layers': [int], # 每个病灶最大层面所在层数，数值从0开始
            'lesion_ranges': [[min, max]], # 每个病灶最大层面所在层数范围，左闭右闭区间，数值从0开始
            'lesion_volumes(cm^3)': [float], # 病灶体积
            'lesions_bm_scores': [float], # 病灶恶性概率
            'lesions_bm_cls': [int], # 病灶良恶性类别，0表示良性，1表示恶性
            'patient_bm_score': float, # 病人恶性概率
            'patient_bm_cls': int, # 病人良恶性类别，0表示良性，1表示恶性
            'ProstateMaskPath': str, # 前列腺腺体分割mask保存路径
            'LesionMaskPath': str, # 前列腺病灶分割mask保存路径
        }

        """
        logging.info('---> Enter prostate inference')
        logging.info('Input parameters: dicom_path=%s', file_paths.get('T2', file_paths.get('T2fs', '')))
        self.result_dict, self.image_dict = {}, {}

        logging.info('Loading images...')
        load_data(file_paths, self.image_dict, config.REGISTER_TARGET)

        if not is_registered:
            self._register()

        logging.info('Running prostate coarse segmentation.')
        self._run_coarse_segmentation(exist_prosatate_mask)

        logging.info('Running prostate cropping.')
        self._crop_prostate()

        if self.settings['do_lesion_segmentation']:
            logging.info('Running lesion segmentation.')
            self._run_lesion_segmentation(exist_lesion_mask)

        if self.settings['do_lesion_bm_classification']:
            logging.info('Running lesion Classification of benign and malignant lesions.')
            self._run_lesion_bm_cls()

        if save_dir:
            logging.info('Saving results to file: %s', save_dir)
            self._save_result(save_dir, save_name)
            # save_raw_image(self.image_dict['raw_images_sitk'], save_dir)

        return self.result_dict

    @staticmethod
    def log_config(config_path):
        with open(config_path, 'rt') as cf:
            config_str = '\n' + cf.read()
            logging.info(config_str)

    @staticmethod
    def setup_logging_system(log_config):
        with open(log_config, 'rt') as f:
            config = yaml.safe_load(f.read())
            logging.config.dictConfig(config)
