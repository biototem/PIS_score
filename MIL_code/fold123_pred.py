import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
import math
from PIL import Image
import cv2
import imageio
import numpy as np
import tiffslide
import torch
import random_seed
from t_mil.model_tmil import T_MIL
import matplotlib.pyplot as plt
random_seed.seed_everything(2023)

def load_model(in_weight):
  ck = torch.load(in_weight, 'cpu')
  ck = ck['state_dict']
  out_ck = {}
  for k, v in ck.items():
    k = k.removeprefix('backbone.')
    if k == 'inst_loss_func.labels':
      continue
    if k == 'instance_loss_fn.labels':
      continue
    out_ck[k] = v

  net = T_MIL(feat_dim=1536, n_classes=2, latent_dim=512, num_heads=8, architecture='LA_MIL', max_limit_patch=15000)
  net.load_state_dict(out_ck)
  net.eval()
  net.cuda()
  return net


in_weight_fold1 = '/media/totem_new2/totem/HIS_data/output/iBOTViT/la_mil/1/best-epoch=082-val_f1=0.7345.ckpt'
in_weight_fold2 = '/media/totem_new2/totem/HIS_data/output/iBOTViT/la_mil/2/best-epoch=096-val_f1=0.6880.ckpt'
in_weight_fold3 = '/media/totem_new2/totem/HIS_data/output/iBOTViT/la_mil/3/best-epoch=008-val_f1=0.7457.ckpt'
fold1_net = load_model(in_weight_fold1)
fold2_net = load_model(in_weight_fold2)
fold3_net = load_model(in_weight_fold3)


list_test_tmp = ['C3L-00001-21.svs','C3L-00009-21.svs','C3L-00080-21.svs']
feat_LocationBlockTable_dir ='/media/totem_new2/totem/HIS_data/CPTAC_LUAD_his/位置信息_224/'
feat_dir = '/media/totem_new2/totem/HIS_data/CPTAC_LUAD_his/iBOTViT_224/'

for file_name in list_test_tmp:
  name = os.path.splitext(file_name)[0]
  cs_batch = torch.as_tensor(np.load(feat_LocationBlockTable_dir+name+'.npy')[:,:4],dtype=torch.float32).cuda()
  feat_batch = torch.as_tensor(np.load(feat_dir+name+'.npy')[:,0,:],dtype=torch.float32).cuda()

  with torch.no_grad():
    output_tmp = (fold1_net([feat_batch], [cs_batch]))  # 结果---lamil
    output_tmp33 = (output_tmp).softmax(-1).cpu().numpy()
    fold1_pred_1_tmp = output_tmp33[0, 1]

    output_tmp = (fold2_net([feat_batch], [cs_batch]))  # 结果---lamil
    output_tmp33 = (output_tmp).softmax(-1).cpu().numpy()
    fold2_pred_1_tmp = output_tmp33[0, 1]

    output_tmp = (fold3_net([feat_batch], [cs_batch]))  # 结果---lamil
    output_tmp33 = (output_tmp).softmax(-1).cpu().numpy()
    fold3_pred_1_tmp = output_tmp33[0, 1]

    pred_1_tmp = (fold1_pred_1_tmp+fold2_pred_1_tmp+fold3_pred_1_tmp)/3
    print(file_name,pred_1_tmp)