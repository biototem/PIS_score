import time,os,torch
from Feature_extraction.utils.openslide_utils import Slide as Open_Slide
import numpy as np
from torch.utils.data import Dataset,DataLoader
from tqdm import tqdm



class roi_dataset(Dataset):
    def __init__(self,img_list):
        super().__init__()
        self.images_lst = img_list
        self.Cache_slide = {}
    def __len__(self):
        return len(self.images_lst)
    def __getitem__(self, idx):
        index_n  = self.images_lst[idx]
        wsi_path_tmp = index_n[-1]
        if wsi_path_tmp not in self.Cache_slide:
            img_slide = Open_Slide(wsi_path_tmp)
            self.Cache_slide[wsi_path_tmp] = img_slide
        img_slide = self.Cache_slide[wsi_path_tmp]
        最靠近目标分辨率的采样层级 = index_n[4]
        index_20x_h1,index_20x_w1,index_20x_h2,index_20x_w2,index_20x_size = index_n[0],index_n[1],index_n[2],index_n[3],index_n[5]
        h_10x = round(index_20x_h1 - ((index_20x_h2 - index_20x_h1) / 2))
        w_10x = round(index_20x_w1 - ((index_20x_w2 - index_20x_w1) / 2))
        img_10x_size = index_20x_size * 2
        image_10x = img_slide.read_region((w_10x,h_10x), 最靠近目标分辨率的采样层级, (img_10x_size,img_10x_size))
        image_10x = image_10x.convert('RGB')
        image_10x_ys = np.array(image_10x, dtype=np.uint8)
        tmp_1um_75 = index_n[6]
        if not tmp_1um_75:
            image_10x_ys = np.where(np.all(image_10x_ys == 0, axis=2, keepdims=True), 255, image_10x_ys)
        tmppp_A = round((index_20x_size / 2))
        tmppp_B = round(index_20x_size+(index_20x_size / 2))
        image_10x_11 =  torch.as_tensor(image_10x_ys)
        return image_10x_11,torch.tensor([tmppp_A,tmppp_B],dtype=torch.int16)


def fun_read(q1,batch_size,wsi_path_list,patch_ref_dir_20x,output_feat_dir,stop_file):
    for wsi_path in wsi_path_list:
        file_name = os.path.basename(wsi_path)
        name_tmp2 = file_name.replace('.tif','').replace('.svs','')
        # if name_tmp2+'.npy' in os.listdir(output_feat_dir):continue
        # if name_tmp2+'.npy' not  in os.listdir('/media/totem_disk/totem/hebingdou/0_5_23_TMP/SYSUCC_Fengzi/位置信息_224_ys/'):continue
        # file = open(output_feat_dir + name_tmp2 +'.npy', 'w')
        title_index_n = np.load(patch_ref_dir_20x  +name_tmp2+'.npy')
        list_index_n = [ list(i) +[wsi_path] for i in title_index_n]
        database_loader = DataLoader(roi_dataset(list_index_n), batch_size=batch_size, num_workers=int(batch_size/2), shuffle=False, pin_memory=True)
        for batch_10X,index_1 in tqdm(database_loader):
            q1.put([True, batch_10X.cuda(),index_1,name_tmp2])
    while True:
        if os.path.exists(stop_file):
            print('停止预测进程1')
            break
        else:
            q1.put([False])
        time.sleep(2)









