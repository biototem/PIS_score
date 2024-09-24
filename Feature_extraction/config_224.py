import os
from PIL import Image
Image.MAX_IMAGE_PIXELS=70746520960
from pathlib import Path
def get_npy_file_paths(directory):
    directory_path = Path(directory)
    npy_file_paths = sorted([str(file_path) for file_path in directory_path.rglob('*.svs')])
    return npy_file_paths


target_size = 224
target_mpp = 0.5

data_type = 'train'
# data_type = 'test'

feat_model_path = '/media/totem_new2/totem/model_jit/iBOTViT_jit_224.pth'
his_target_path = "13#24.png"
wsi_dir = '/media/totem_kingston/totem/tmp_svs/'
mask_png_dir =  '/media/totem_new2/totem/HIS_data/mask/'
wsi_path_list = get_npy_file_paths(wsi_dir)


out_dir = '/media/totem_kingston/totem/位置信息/'
output_feat_xy_dir = out_dir+ '/位置信息_224/'
output_feat_dir = out_dir+'/iBOTViT/'
output_feat_xy_dir_可视化 = out_dir+'/weiz_v/'
os.makedirs(os.path.dirname(output_feat_xy_dir + '/'), exist_ok=True)
os.makedirs(os.path.dirname(output_feat_xy_dir_可视化 + '/'), exist_ok=True)
os.makedirs(os.path.dirname(output_feat_dir + '/'), exist_ok=True)

