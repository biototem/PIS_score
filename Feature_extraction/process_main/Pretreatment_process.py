import time
import numpy as np
import torch

def fun_Pretreatment(q2,feat_model_path,output_feat_dir,data_type,stop_file):
    model1 = torch.jit.load(feat_model_path, 'cpu')
    model1.cuda()
    model1.eval()
    oooo_tmp_name = ''
    new_t_img_list = []
    with torch.no_grad():
        while True:
            if not q2.empty():
                tmp_data = q2.get()
                if not tmp_data[0]:
                    if len(new_t_img_list)>0:
                        new_transform_img = (torch.cat(new_t_img_list)).cpu().numpy()
                        np.save(output_feat_dir + oooo_tmp_name + '.npy', new_transform_img)
                    break
                if data_type == 'train':
                    feat_tmp_tmp_20x = model1(tmp_data[1])
                    feat_tmp_tmp_10x = model1(tmp_data[2])
                    feat_tmp_tmp_20x_增强 = model1(tmp_data[3])
                    feat_tmp_tmp_10x_增强 = model1(tmp_data[4])

                    # _,feat_tmp_tmp_20x = model1(tmp_data[1])
                    # _,feat_tmp_tmp_10x = model1(tmp_data[2])
                    # _,feat_tmp_tmp_20x_增强 = model1(tmp_data[3])
                    # _,feat_tmp_tmp_10x_增强 = model1(tmp_data[4])
                    all_10_feat = []
                    feat_tmp_tmp_ys = torch.cat([feat_tmp_tmp_20x, feat_tmp_tmp_10x], dim=1)
                    all_10_feat.append(feat_tmp_tmp_ys)
                    feat_tmp_tmp_增强 = torch.cat([feat_tmp_tmp_20x_增强, feat_tmp_tmp_10x_增强], dim=1)
                    all_10_feat.append(feat_tmp_tmp_增强)
                    feat_tmp = torch.stack(all_10_feat,dim=1)
                else:
                    feat_tmp_tmp_20x = model1(tmp_data[1])
                    feat_tmp_tmp_10x = model1(tmp_data[2])
                    feat_tmp = torch.cat([feat_tmp_tmp_20x, feat_tmp_tmp_10x], dim=1)
                if (oooo_tmp_name!=tmp_data[5]):
                    if oooo_tmp_name != '':
                        new_transform_img = (torch.cat(new_t_img_list)).cpu().numpy()
                        np.save(output_feat_dir + oooo_tmp_name + '.npy', new_transform_img)
                        new_t_img_list = []
                    oooo_tmp_name = tmp_data[5]
                new_t_img_list.append(feat_tmp)
    time.sleep(2)
    file = open(stop_file, 'w')  # 不能去除,作用是帮助结束进程
