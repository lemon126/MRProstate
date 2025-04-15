import os.path
import numpy as np
import SimpleITK as sitk

from batchgenerators.utilities.file_and_folder_operations import *
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIO


class Segmentor(object):
    def __init__(self, model_path, device, use_mirroring=False):
        """
        load model.
        :param model_path: 'str'. Model path, like '/../../model_final_checkpoint.model'. The configuration file
        'model_final_checkpoint.model.pkl' must be in the same directory.
        :param model_name: Model name, used for model selection. The default is None, corresponding to the native nnUNet
        model.
        """
        predictor = nnUNetPredictor(
            tile_step_size=0.5,
            use_gaussian=True,
            use_mirroring=use_mirroring,
            perform_everything_on_device=True,
            device=device,
            verbose=False,
            verbose_preprocessing=False,
            allow_tqdm=True
        )
        checkpoint_name = os.path.basename(model_path)
        fold = os.path.basename(os.path.dirname(model_path)).replace('fold_', '')
        model_dir = os.path.dirname(os.path.dirname(model_path))
        predictor.initialize_from_trained_model_folder(
            model_dir,
            use_folds=(fold),
            checkpoint_name=checkpoint_name,
        )
        self.predictor = predictor

    def run_inference(self, image_paths_or_array, props=None, save_path=None):
        """run nnunet inference
        :param image_paths_or_array: Input a group of image paths, such as [../t2.nii.gz, ../adc.nii.gz, ../dwi.nii.gz],
         or input the image data that has already been read in the form of [C, D, H, W], or input a single data point in
         the format of SimpleITK.Image.
        :param props: When the input is an image array, additional image information must be provided.
        ：param save_path: Result saving path.

        return mask_sitk: Return mask in the format of SimpleITK Image.

        """
        if isinstance(image_paths_or_array, Union[list, tuple]):
            image_arr, props = SimpleITKIO().read_images(image_paths_or_array)
        elif isinstance(image_paths_or_array, np.ndarray):
            image_arr = image_paths_or_array.astype(np.float32)
            assert props is not None, '输入为图像数组时，属性参数props不能为空。'
        elif isinstance(image_paths_or_array, sitk.Image):
            image_arr = [sitk.GetArrayFromImage(image_paths_or_array).astype(np.float32)]
            props = {
                'sitk_stuff': {
                    # this saves the sitk geometry information. This part is NOT used by nnU-Net!
                    'spacing': image_paths_or_array.GetSpacing(),
                    'origin': image_paths_or_array.GetOrigin(),
                    'direction': image_paths_or_array.GetDirection()
                },
                # the spacing is inverted with [::-1] because sitk returns the spacing in the wrong order lol. Image arrays
                # are returned x,y,z but spacing is returned z,y,x. Duh.
                'spacing': image_paths_or_array.GetSpacing()[::-1]
            }
        else:
            raise ValueError('The input types only support list, np.ndarray, and SimpleITK.Image.')

        ret = self.predictor.predict_single_npy_array(image_arr, props, None, None, False)

        mask_sitk = sitk.GetImageFromArray(ret)
        mask_sitk.SetOrigin(props['sitk_stuff']['origin'])
        mask_sitk.SetDirection(props['sitk_stuff']['direction'])
        mask_sitk.SetSpacing(props['sitk_stuff']['spacing'])
        if save_path:
            sitk.WriteImage(mask_sitk, save_path)

        return mask_sitk
