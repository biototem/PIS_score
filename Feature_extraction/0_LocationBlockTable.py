import os
from Feature_extraction.utils.read_get_index import get_index_list
import numpy as np
import imageio
import  multiprocessing  as mul
import config_224 as config

def fun(wsi_path):
    try:
        file_name = os.path.basename(wsi_path)
        name, ext = os.path.splitext(file_name)
        mask_01_array =  imageio.v3.imread(config.mask_png_dir + name+'.png')
        list_0_tmp = get_index_list(wsi_path, config.target_mpp, config.target_size, mask_01_array)
        HW_index_array = np.array(list_0_tmp)
        np.save(config.output_feat_xy_dir +'/'+ name +   '.npy',HW_index_array)
    except:
        print(wsi_path)

if __name__ == '__main__':
    pool = mul.Pool(10)
    pool.map(fun,config.wsi_path_list)
