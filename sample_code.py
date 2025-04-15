from api import MRProstate

file_paths = {
     'ADC': '/path/to/ADC_folder/',  # or 'ADC': '/path/to/ADC.nii.gz'
     'DWI': '/path/to/DWI_folder/',  # or 'DWI1000': '/path/to/ADC.nii', ...
     'T2': '/path/to/T2_folder/',    # or 'T2': '/path/to/T2.nii.gz'
     # 'T2fs'、'DWI1000', 'DWI2000', 'DWI3000'...
}
save_dir = './result'
prostate = MRProstate()
result = prostate.run(file_paths, save_dir, save_name='')
print(result)
