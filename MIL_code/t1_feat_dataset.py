import random
import numpy as np
from torch.utils.data.dataset import Dataset
import torch


class FeatDataset(Dataset):
    def __init__(self, in_feat_0_list,in_feat_1_list, cls_blance, batch_count, enchance=False,in_patch_ref_dir=None,feat_dir = None):
        self.cls_blance = cls_blance
        self.batch_count = batch_count
        self.enchance = enchance
        self.in_patch_ref_dir = in_patch_ref_dir
        self.feat_dir = feat_dir
        self.has_coords = in_patch_ref_dir is not None
        if cls_blance :

            in_feat_0_count = len(in_feat_0_list)
            in_feat_1_count = len(in_feat_1_list)

            if in_feat_0_count > in_feat_1_count:
                # 指定要采样的数量
                sample_size = in_feat_0_count - in_feat_1_count
                # 随机采样并保存生成的列表
                for iiiu in range(sample_size):
                    # 随机采样并保存生成的列表
                    sampled_list = random.choice(in_feat_1_list)
                    in_feat_1_list.append(sampled_list)
            elif in_feat_0_count < in_feat_1_count:
                # 指定要采样的数量
                sample_size = in_feat_1_count - in_feat_0_count
                # 随机采样并保存生成的列表
                for iiiu1 in range(sample_size):
                    # 随机采样并保存生成的列表
                    sampled_list1 = random.choice(in_feat_0_list)
                    in_feat_0_list.append(sampled_list1)
            in_feat_cls_list_new_tmp = in_feat_0_list + in_feat_1_list
            in_feat_cls_list_new = []
            for rtjt in in_feat_cls_list_new_tmp:
                rtjt_tmp_ys = [rtjt[0],rtjt[1],0]
                rtjt_tmp_new = [rtjt[0], rtjt[1],1]
                in_feat_cls_list_new.append(rtjt_tmp_ys)
                in_feat_cls_list_new.append(rtjt_tmp_new)
        else:
            in_feat_cls_list_new_tmp = in_feat_0_list + in_feat_1_list
            in_feat_cls_list_new = []
            for rtjt in in_feat_cls_list_new_tmp:
                rtjt_tmp_ys = [rtjt[0],rtjt[1],0]
                in_feat_cls_list_new.append(rtjt_tmp_ys)
        random.shuffle(in_feat_cls_list_new)
        self.items = in_feat_cls_list_new
    def __len__(self):
        if self.batch_count is None:
            return len(self.items)
        else:
            return self.batch_count
    def __getitem__(self, item):
        if self.batch_count is not None:
            if item >= self.batch_count:
                raise StopIteration

        if self.cls_blance:
            if self.batch_count is None and item > len(self.items):
                raise StopIteration

        feat_file, cls,_ = self.items[item][0],self.items[item][1],self.items[item][2]

        feat_file1 = self.feat_dir  +str(feat_file).replace('.svs','.npy')
        feat = np.load(feat_file1, mmap_mode='c', allow_pickle=False)

        coords = None
        if self.has_coords:
            patch_ref_file = self.in_patch_ref_dir +'/'+ str(feat_file).replace('.svs','.npy')
            coords = np.load(patch_ref_file, mmap_mode='c')[:,:4]


        # print('---------------',feat.shape,coords.shape)
        if self.enchance:
            ids1 = np.arange(feat.shape[0], dtype=np.int32)
            ids2 = np.random.randint(feat.shape[1], size=feat.shape[0], dtype=np.int32)
            feat = feat[ids1, ids2]
        else:
            feat = feat[:, 0]
        if self.enchance:
            # drop
            if np.random.uniform() < 0.8:
                # TODO move keep_n_param to cfg
                # print('TODO move keep_n_param to cfg')
                # new
                # keep_n = int(np.random.uniform(0.8, 1.) * feat.shape[0])
                # ori
                keep_n = int(np.random.uniform(0.3, 1.) * feat.shape[0])
                keep_n = max(2, keep_n)
                ids = np.arange(feat.shape[0])
                np.random.shuffle(ids)
                ids = ids[:keep_n]
                feat = feat[ids]
                if coords is not None:
                    coords = coords[ids]
                    assert len(feat) == len(coords)
            # noise
            noise = np.random.normal(size=feat.shape, scale=feat.std()*np.random.uniform(0.01, 0.2)).astype(feat.dtype)
            feat += noise

        if coords is not None:
            coords = torch.from_numpy(coords)

        r = (feat, cls,coords,None)


        return r



if __name__ == '__main__':
    import os

    feat_0 = '/mnt/totem-bak/totem/hebingdou_data/feats_dir_copy/feats_dir_2/data/feat/1/train/0/'
    feat_1 = '/mnt/totem-bak/totem/hebingdou_data/feats_dir_copy/feats_dir_2/data/feat/1/train/1/'

    kkk_0 = []
    kkk_1 = []

    for i in os.listdir(feat_0):
        kkk_0.append([feat_0+i,0])
    for i in os.listdir(feat_1):
        kkk_1.append([feat_1+i,1])


    ds = FeatDataset(kkk_0,kkk_1, cls_blance=False, batch_count=600, enchance=False)

    batchcount = 300
    batchsize = 2

    for feat_i, (feat, cls) in enumerate(ds):
        print(cls)

