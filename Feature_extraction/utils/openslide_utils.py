# @Time    : 2018.10.9
# @Author  : kawa Yeung
# @Licence : bio-totem

import os
import openslide
from PIL import Image
import numpy as np


class Slide(openslide.OpenSlide):
    def __init__(self, svs_file, level=2):
        """
        open svs file with open-slide
        :param svs_file: svs file, absolute path
        :return: slide
        """
        super().__init__(svs_file)
        self._filepath = svs_file
        self._basename = os.path.basename(svs_file).split('.')[0]
        self.slide = openslide.OpenSlide(svs_file)
        self._level = level

    def get_basename(self):
        """
        return svs file basename, not contain file suffix
        :return:
        """

        return self._basename

    def get_filepath(self):
        """
        get absolute svs file
        :return:
        """

        return self._filepath

    def get_level(self):
        """
        return level
        :return:
        """

        return self._level

    def get_level_downsample(self, level=2):
        """
        get the expected level downsample, default level two
        :param level: level, default 2
        :return: the level downsample
        """

        return self.slide.level_downsamples[level]

    def get_level_dimension(self, level=2):
        """
        get the expected level dimension, default level two
        :param level: level, default 0
        :return:
        """

        return self.slide.level_dimensions[level]

    def get_thumb(self, level=2):
        """
        get thumb image
        :return:
        """

        level_dimension = self.get_level_dimension(level)
        tile = self.slide.get_thumbnail(level_dimension)

        return tile
    def get_最靠近指定分辨率下的层级(self, target_mpp):
        最靠近目标分辨率下的层级 = 0
        jfjfp_pfpf = 99999
        min_img_lever = self.slide.level_count
        for tmp_i in range(min_img_lever):
            try:

                mpp_in_level = self.properties['tiffslide.mpp-x'] * self.slide.level_downsamples[tmp_i]
            except:
                mpp_in_level = 0.5 * self.slide.level_downsamples[tmp_i]
            ooooo_tmp = abs(target_mpp - mpp_in_level)
            if ooooo_tmp < jfjfp_pfpf:
                jfjfp_pfpf = ooooo_tmp
                最靠近目标分辨率下的层级 = tmp_i
        return 最靠近目标分辨率下的层级

    def svs_to_png(self,save_dir):
        """
        convert svs to png
        :return:
        """
        self.get_thumb().save(save_dir)

    def expand_img(self, im, size, value=(0, 0, 0)):
        """
        expand the image
        :param im: the image want to expand
        :param size: tuple, the size of expand
        :param value: tuple, the pixel value at the expand region
        :return: the expanded image
        """

        im_new = Image.new("RGB", size, value)
        im_new.paste(im, (0, 0))

        return im_new
    
    def get_mpp(self):
        """
        get the value of mpp 
        :return: 0.00025
        """
        properties = self.properties
        properties['openslide.mpp-x']
        return float(properties['openslide.mpp-x'])/1000

    # def __del__(self):
    #     self.slide.close()


