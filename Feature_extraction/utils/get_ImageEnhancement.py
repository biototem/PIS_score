import warnings
warnings.filterwarnings("ignore")
from PIL import Image
import torch.nn as nn
import torch
from torchvision import transforms
import albumentations as A


imgto_tensor = transforms.ToTensor()
trnsfrms_val = transforms.Compose(
    [
        transforms.Resize((256,256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ]
)

class get_ImageEnhancement_n_4_20(nn.Module):
    def __init__(self):
        super().__init__()
        def ImageEnhancement_n(image_x):
            size_tmp_H, size_tmp_W = (256, 256)
            ImageEnhancement_1 = A.Compose( [A.Resize(size_tmp_H, size_tmp_W), A.RandomBrightnessContrast(p=1), A.Normalize()])
            ImageEnhancement_2 = A.Compose([A.Resize(size_tmp_H, size_tmp_W), A.RandomGamma(p=1), A.Normalize()])
            ImageEnhancement_3 = A.Compose([A.Resize(size_tmp_H, size_tmp_W), A.ColorJitter(0.1, 0.05, 0.02, 0.02, p=1), A.Normalize()])
            ImageEnhancement_4 = A.Compose( [A.Resize(size_tmp_H, size_tmp_W), A.ImageCompression(85, 100, p=1), A.Normalize()])
            ImageEnhancement_5 = A.Compose([A.Resize(size_tmp_H, size_tmp_W), A.Blur(p=1), A.Normalize()])
            ImageEnhancement_6 = A.Compose([A.Resize(size_tmp_H, size_tmp_W), A.CLAHE(p=1), A.Normalize()])
            ImageEnhancement_7 = A.Compose([A.Resize(size_tmp_H, size_tmp_W), A.RGBShift(p=1), A.Normalize()])
            ImageEnhancement_8 = A.Compose([A.Resize(size_tmp_H, size_tmp_W), A.HueSaturationValue(p=1), A.Normalize()])
            ImageEnhancement_9 = A.Compose([A.Resize(size_tmp_H, size_tmp_W), A.RandomBrightness(p=1), A.Normalize()])

            transformed_image1 = imgto_tensor(ImageEnhancement_1(image=image_x)["image"])
            transformed_image2 = imgto_tensor(ImageEnhancement_2(image=image_x)["image"])
            transformed_image3 = imgto_tensor(ImageEnhancement_3(image=image_x)["image"])

            transformed_image4 = imgto_tensor(ImageEnhancement_4(image=image_x)["image"])
            transformed_image5 = imgto_tensor(ImageEnhancement_5(image=image_x)["image"])
            transformed_image6 = imgto_tensor(ImageEnhancement_6(image=image_x)["image"])

            transformed_image7 = imgto_tensor(ImageEnhancement_7(image=image_x)["image"])
            transformed_image8 = imgto_tensor(ImageEnhancement_8(image=image_x)["image"])
            transformed_image9 = imgto_tensor(ImageEnhancement_9(image=image_x)["image"])

            img_list_tmp = [trnsfrms_val(Image.fromarray(image_x)), transformed_image1, transformed_image2,
                            transformed_image3, transformed_image4, transformed_image5, transformed_image6,
                            transformed_image7,
                            transformed_image8, transformed_image9]
            new_img_nnn = torch.stack(img_list_tmp)
            return new_img_nnn
        self.ImageEnhancement_n_model = ImageEnhancement_n
    def forward(self, tmp_img222):
        tutng = self.ImageEnhancement_n_model(tmp_img222)
        return tutng

