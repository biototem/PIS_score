import os
import numpy as np
import config_224
from Feature_extraction.utils.tiffslide_utils import Slide
import cv2
import imageio


for wsi_path in config_224.wsi_path_list:
    file_name = os.path.basename(wsi_path)
    name, ext = os.path.splitext(file_name)
    img_slide = Slide(wsi_path)
    get_wsi_level = 0
    for tmp1 in range(6):
        if (min(img_slide.get_level_dimension(get_wsi_level)) > 7000):
            get_wsi_level = get_wsi_level + 1
    lever_downsamples = img_slide.level_downsamples[get_wsi_level]
    img_RGB = img_slide.read_region((0, 0), get_wsi_level, img_slide.get_level_dimension(get_wsi_level))
    img_RGB = img_RGB.convert('RGB')
    img_RGB = np.array(img_RGB, dtype=np.uint8)
    xy_array = np.load(config_224.output_feat_xy_dir +'/' + name +'.npy')
    for index in xy_array:
        y1, x1 ,y2, x2= int(index[0]), int(index[1]), int(index[2]), int(index[3])
        title_75 = index[6]
        point1 = (int(x1/lever_downsamples),int(y1/lever_downsamples))
        point2 = (int(x2/lever_downsamples),int(y2/lever_downsamples))
        # #############在图像上绘制矩形框
        if title_75:
            cv2.rectangle(img_RGB, point1, point2, (0, 0, 255), 2)  # (0, 0, 255)表示BGR颜色值，2表示线条宽度
        else:
            cv2.rectangle(img_RGB, point1, point2, (255, 0, 255), 2)  # (0, 0, 255)表示BGR颜色值，2表示线条宽度
    imageio.imwrite(config_224.output_feat_xy_dir_可视化 + name + '.jpg',img_RGB)
