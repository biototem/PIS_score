import torch.backends.cuda as cuda
cuda.benchmark = False
cuda.deterministic = True
import  multiprocessing  as mul
from process_main.read_process import fun_read
from process_main.his_process import fun_his
from process_main.Pretreatment_process import fun_Pretreatment
import time,os
import config_224
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

if __name__ == '__main__':
    batch_size = 8
    stop_file = os.getcwd() + '/process_main/close_tmp_main0.txt'    #因为一些特殊原因，暂时不清楚tensor向量传进队列会需要一直传输，需要用一个东西保证进程结束
    target_mpp = config_224.target_mpp
    target_size = config_224.target_size
    wsi_path_list = config_224.wsi_path_list
    feat_model_path = config_224.feat_model_path
    his_target_path = config_224.his_target_path
    feat_model_name = os.path.basename(config_224.feat_model_path)
    patch_ref_dir_20x = config_224.output_feat_xy_dir
    output_feat_dir = config_224.output_feat_dir
    data_type = config_224.data_type
    cc = time.time()
    q1 = mul.Queue(100) #数据输入队列
    q2 = mul.Queue(100) #数据输入队列
    try:
        os.remove(stop_file)
    except:
        pass
    p0_get_data = mul.Process(target=fun_read, args=(q1,batch_size,wsi_path_list,patch_ref_dir_20x,output_feat_dir,stop_file))
    p0_get_data.start()
    p1_his_data = mul.Process(target=fun_his, args=(q1,q2,feat_model_name,his_target_path,data_type,stop_file))
    p1_his_data.start()
    p2_Pretreatment_data = mul.Process(target=fun_Pretreatment, args=(q2,feat_model_path,output_feat_dir,data_type,stop_file))
    p2_Pretreatment_data.start()
    p2_Pretreatment_data.join()
    p1_his_data.join()
    p0_get_data.join()
    dd = time.time()
    print('Total_running_time: ',dd-cc)




