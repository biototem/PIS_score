import torch
import os,time
from torchvision import transforms
import numpy as np
from Feature_extraction.utils.HIS_normalization_GPU import  read_target,stain_unmixing_routine1,his_normalization_pytorch

from PIL import Image
from Feature_extraction.utils.MIL_图块增强 import MyAugFunc


transforms_dino_vit_small_patch16_ep200_jit = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean= ( 0.70322989, 0.53606487, 0.66096631 ), std= ( 0.21716536, 0.26081574, 0.20723464 ))
        ])


transforms_PathoDuet_beit = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()]
)
transforms_beit3 = transforms.Compose([
    transforms.Resize((224,224), interpolation=3),
    transforms.ToTensor(),
    transforms.Normalize(mean= (0.5, 0.5, 0.5), std= (0.5, 0.5, 0.5))
        ])


trnsfrms_CTransPath_Brow_iBOTViT_UNI = transforms.Compose(
    [
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean = (0.485, 0.456, 0.406), std =  (0.229, 0.224, 0.225))
    ]
)
trnsfrms_val_RetCCL = transforms.Compose([
        transforms.Resize((256,256)),
        transforms.ToTensor(),
        transforms.Normalize(mean = (0.485, 0.456, 0.406), std = (0.229, 0.224, 0.225))])


def fun_his(q1,q2,feat_model_name,im_target_path,data_type,stop_file):
    im_target = torch.tensor(read_target(im_target_path)).cuda()
    W_target1 = stain_unmixing_routine1(im_target, stain_color_map_hematoxylin=torch.tensor([0.65, 0.70, 0.29]).cuda()).cuda()
    his_model = his_normalization_pytorch().cuda()

    aug_func = MyAugFunc()
    print(feat_model_name)
    if (feat_model_name == 'dino_vit_s16_jit_224.pth') | (feat_model_name == 'dino_vit_s8_jit_224.pth'):
        trnsfrms_val = transforms_dino_vit_small_patch16_ep200_jit
    elif (feat_model_name == 'beit3_jit_224.pth'):
        trnsfrms_val = transforms_beit3
    elif (feat_model_name == 'PathoDuet_jit_224.pth') | (feat_model_name == 'beit_jit_224.pth'):
        trnsfrms_val = transforms_PathoDuet_beit
    elif (feat_model_name == 'iBOTViT_jit_224.pth') | (feat_model_name == 'CTransPath_jit_224.pth') | (feat_model_name == 'BROW_jit_224.pth') | (feat_model_name == 'UNI_jit_224.pth'):
        trnsfrms_val = trnsfrms_CTransPath_Brow_iBOTViT_UNI
    elif feat_model_name == 'RetCCL_jit_256.pth':
        trnsfrms_val = trnsfrms_val_RetCCL

    with torch.no_grad():
        while True:
            if not q1.empty():
                tmp_data = q1.get()
                if not tmp_data[0]:
                    q2.put([False])
                    break
                data_10x = tmp_data[1]
                tmppp_A,tmppp_B = tmp_data[2][0][0],tmp_data[2][0][1]
                file_name123 = tmp_data[3]
                data_10x_out =  his_model(data_10x, W_target1)
                title_10x_11_list = []
                title_20x_11_list = []
                title_10x_22_list = []
                title_20x_22_list = []
                for iii2 in range(data_10x_out.shape[0]):
                    title_10x = data_10x_out[iii2,:,:,:]
                    title_10x = ((title_10x).cpu().detach()).numpy().astype(np.uint8).transpose((1,2,0))
                    title_20x = title_10x[tmppp_A:tmppp_B,tmppp_A:tmppp_B]
                    title_10x_11 = trnsfrms_val(Image.fromarray(title_10x))
                    title_20x_11 = trnsfrms_val(Image.fromarray(title_20x))
                    title_10x_11_list.append(title_10x_11)
                    title_20x_11_list.append(title_20x_11)

                    if data_type == 'train':
                        image_10x_增强 = aug_func(title_10x)
                        image_10x_增强_20x = image_10x_增强[tmppp_A:tmppp_B,tmppp_A:tmppp_B]
                        title_10x_22 = trnsfrms_val(Image.fromarray(image_10x_增强))
                        title_20x_22 = trnsfrms_val(Image.fromarray(image_10x_增强_20x))
                        title_10x_22_list.append(title_10x_22)
                        title_20x_22_list.append(title_20x_22)
                if data_type == 'train':
                    q2.put([True,torch.stack(title_20x_11_list).cuda(),torch.stack(title_10x_11_list).cuda(),torch.stack(title_20x_22_list).cuda(),torch.stack(title_10x_22_list).cuda(), file_name123])
                else:
                    q2.put([True,torch.stack(title_20x_11_list).cuda(),torch.stack(title_10x_11_list).cuda(),'', '', file_name123])
    while True:
        if os.path.exists(stop_file):
            print('停止预测进程2')
            break
        else:
            q2.put([False])
        time.sleep(2)







