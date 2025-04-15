# coding=utf-8
from yacs.config import CfgNode as CN

# from lesion_cls.default_configs import DefaultConfigs


class configs():
    # ============ model setting ============
    model = 'cls/hardnet_2'
    loss_weight = [0.35, 0.65]

    input_channel = 8
    num_cls_classes = 2
    input_size = (16, 128, 128)
    target_spacing_zyx = [3.0, 0.390625, 0.390625]

    # HardNet配置
    HARDNET = CN()
    HARDNET.ARCH = 39

    test_modalities = ['T2', 'T2fs', 'ADC', 'DWI1000', 'DWI1500', 'DWI2000', 'DWI3000', 'lesionMask']

    def __init__(self):
        super(configs, self).__init__()
        pass
