import gc
import torch
import kornia as K  ################pip install kornia
from kornia.augmentation import RandomBoxBlur,RandomGaussianNoise,RandomAffine,RandomElasticTransform
import torch.nn as nn


class MyAugmentation(nn.Module):
    def __init__(self):
        super().__init__()

        _augmentations = nn.Sequential(
            K.augmentation.RandomHorizontalFlip(p=0.75),
            K.augmentation.RandomVerticalFlip(p=0.75),
            K.augmentation.RandomPerspective(0.5, "nearest", align_corners=True, same_on_batch=False, keepdim=False,p=0.5),
            K.augmentation.RandomRotation(15.0, "nearest", align_corners=True, same_on_batch=False, keepdim=False, p=0.5),
            K.augmentation.RandomBrightness(brightness=(0.8, 1.2), clip_output=True, same_on_batch=False, keepdim=False,p=0.5)
        )
        # self.k5 = RandomBoxBlur((21, 5), "reflect", same_on_batch=False, keepdim=False, p=1.0)
        # self.k6 = RandomGaussianNoise(mean=0.2, std=0.7, same_on_batch=False, keepdim=False, p=1.0)
        # self.k7 = RandomAffine((-15.0, 5.0), (0.3, 1.0), (0.4, 1.3), 0.5, resample="nearest",  padding_mode="reflection", align_corners=True, same_on_batch=False, keepdim=False, p=1.0)


    def forward(self, img_tmpp1111: torch.Tensor) -> torch.Tensor:
        img_tmpp1111_ys = img_tmpp1111.detach()
        img_tmpp1111_out1 = self.k1(img_tmpp1111)
        img_tmpp1111_out2 = self.k2(img_tmpp1111)
        feat_tmp_1 = K.enhance.adjust_brightness(img_tmpp1111, torch.linspace(0.2, 0.8, img_tmpp1111.shape[0]))
        feat_tmp_2 = K.enhance.adjust_contrast(img_tmpp1111, torch.linspace(0.5, 1.0, img_tmpp1111.shape[0]))
        feat_tmp_3 = self.k2(feat_tmp_1)
        feat_tmp_4 = self.k2(feat_tmp_2)
        # feat_tmp_5 = self.k5(img_tmpp1111)
        # feat_tmp_6 = self.k6(img_tmpp1111)
        # feat_tmp_7 = self.k7(img_tmpp1111)
        tmp_pffp = [img_tmpp1111_ys,img_tmpp1111_out1,img_tmpp1111_out2,feat_tmp_1,feat_tmp_2]#,feat_tmp_3,feat_tmp_4]#feat_tmp_5,feat_tmp_6,feat_tmp_7]

        new_img_nnn = torch.stack(tmp_pffp)
        try:
            del tmp_pffp,img_tmpp1111,img_tmpp1111_out1,img_tmpp1111_out2,feat_tmp_1,feat_tmp_2#,feat_tmp_3,feat_tmp_4#,feat_tmp_5,feat_tmp_6,feat_tmp_7
            gc.collect()
        except:
            pass
        try:
            del tmp_pffp,img_tmpp1111,img_tmpp1111_out1,img_tmpp1111_out2,feat_tmp_1,feat_tmp_2#,feat_tmp_3,feat_tmp_4#,feat_tmp_5,feat_tmp_6,feat_tmp_7
            gc.collect()
        except:
            pass
        return new_img_nnn












