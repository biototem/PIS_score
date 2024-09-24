import gc
import numpy as np
from hebingdou_utils.asap_slide import Writer
import math
import cv2

def mask_to_tif(img_slide,mask_data_tmp,mask_data_level,output_path):
    try:
        mpp, mag = img_slide.get_mpp_mag_ys()
    except:
        mpp = 0.25
        mag = 40
    dimensions = img_slide.level_dimensions[0]
    mask_level_ds = round(img_slide.level_downsamples[mask_data_level])
    tilesize = 512
    mask_level_tilesize = round(tilesize/mask_level_ds)
    assert mask_level_tilesize in [16,32,64,128,256,512] ,'要求mask层级不能太低'
    w_count = math.ceil(mask_data_tmp.shape[1] / mask_level_tilesize)
    h_count = math.ceil(mask_data_tmp.shape[0] / mask_level_tilesize)
    with Writer(
            output_path=output_path,
            tile_size=tilesize,
            dimensions=dimensions,
            spacing=mpp,
    ) as writer:
        for w in range(w_count):
            for h in range(h_count):
                tmp_img = mask_data_tmp[h * mask_level_tilesize:h * mask_level_tilesize + mask_level_tilesize, w * mask_level_tilesize:w * mask_level_tilesize + mask_level_tilesize]
                if tmp_img.shape!=(mask_level_tilesize,mask_level_tilesize):
                    cur_t1 = np.zeros((mask_level_tilesize, mask_level_tilesize), dtype=np.uint8)
                    cur_t1[0:tmp_img.shape[0], 0:tmp_img.shape[1]] = tmp_img
                else:
                    cur_t1 = tmp_img
                cur_t2 = cv2.resize(cur_t1,(tilesize,tilesize),interpolation=cv2.INTER_NEAREST)
                writer.write(tile=cur_t2, x=w * tilesize, y=h * tilesize)

    try:
        del mask_data_tmp,cur_t2,cur_t1,tmp_img
        gc.collect()
        del mask_data_tmp,cur_t2,cur_t1,tmp_img
        gc.collect()
        del mask_data_tmp,cur_t2,cur_t1,tmp_img
        gc.collect()
        del mask_data_tmp,cur_t2,cur_t1,tmp_img
        gc.collect()
        del mask_data_tmp,cur_t2,cur_t1,tmp_img
        gc.collect()
    except:
        pass
