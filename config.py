import os


base_model_dir = './checkpoints'
MODEL_PATHS = {
    'CoarseSeg': os.path.join(base_model_dir, 'prostate_seg/Dataset610_ProstateSeg202404/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_all/checkpoint_best.pth'),
    'LesionSeg': os.path.join(base_model_dir, 'lesion_seg/Dataset602_ProstateLesionSeg202404/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_all/checkpoint_final.pth'),
    'LesionBMCls': {
        'model_ckpt_path': os.path.join(base_model_dir, 'BM_lesion_cls/model.pth')
    }
}

CONFIG_PATHS = {
    'log': './log.yaml'
}

PROSTATE_SEG_REQUIRED_FIELDS = [['T2', 'T2fs']] # ['T2', 'T2fs']中至少包含一个
PROSTATE_SEG_FIELDS = [['T2fs', 'T2']] # ['T2fs', 'T2']表示优先级从高到低， ‘T2fs’优先

LESION_SEG_REQUIRED_FIELDS = [['T2', 'T2fs'], ['ADC', 'DWI']] # ['T2', 'T2fs']中至少包含一个，['ADC', 'DWI']中至少包含一个
LESION_SEG_FIELDS = [['T2', 'T2fs'], 'ADC', 'DWI']  # ['T2', 'T2fs']表示优先级从高到低， ‘T2’优先

LESION_BM_CLS_REQUIRED_FIELDS = [['T2', 'T2fs'], ['ADC', 'DWI']]
LESION_BM_CLS_FIELDS = ['T2', 'T2fs', 'ADC', 'DWI1000', 'DWI1500', 'DWI2000', 'DWI3000']

REGISTER_TARGET = ['T2fs', 'T2']

CROP_PADDING = {
    'for_lesion_seg': '60',
}
CROP_PADDING_mm = 18
