'''
步骤5 配置文件
'''
import pickle


task = 'single_class'
assert task in ('single_class', 'multi_binary_class', 'regression', 'regression-mse-ord-r03-z200', 'regression-ord-r03-z100')


with open('/media/totem_new2/totem/HIS_data/output/iBOTViT/main_train/fold_train_index.pkl', 'rb') as file:
  all_dice_tmp = pickle.load(file)

xxxxxxx111 = '2'
data_index_n=all_dice_tmp[xxxxxxx111]
in_train_feat_0_list = data_index_n['train_0']
in_train_feat_1_list = data_index_n['train_1']
in_valid_feat_0_list = data_index_n['val_0']
in_valid_feat_1_list = data_index_n['val_1']



model_type = 'la_mil'
assert model_type in ['rtransformer', 'rtransformer_v2', 'clam_sb', 'clam_mb', 'la_mil', 'ga_mil', 'dtfd_mil']

feat_LocationBlockTable_dir ='/media/totem_new2/totem/HIS_data/位置信息_224/'
feat_dir = '/media/totem_kingston/totem/iBOTViT/'

ck_dir = '/media/totem_new2/totem/HIS_data/output/iBOTViT/ibot/'+'/'+model_type+'/'+xxxxxxx111
score_show_file = ck_dir + '/score.txt'

feat_dim = 1536

n_cls = 2 #几分类任务
device = 0
epoch = 400
batchcount = 10
batchsize = 2
n_worker = 4
lr=1e-4
max_n_aug = None






