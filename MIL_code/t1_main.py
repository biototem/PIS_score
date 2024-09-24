'''
步骤5 主要功能
'''
import time

import pytorch_lightning as pl
import os
import numpy as np
import torch
import torch.utils.data
from pytorch_lightning.callbacks import ModelCheckpoint
from t1_feat_dataset import FeatDataset
import warnings
# 忽略所有警告
warnings.filterwarnings("ignore")
# 现在运行代码，警告将不会显示
import random_seed
random_seed.seed_everything(2023)

def my_collate_func(batch: list):
    # xs 输入特征；ys 目标类别；cs 每条特征的坐标, infos 为更多信息
    # 注意，xs 长度不定，ys 长度也不定，cs 长度也不定，所以传递为 list
    xs, ys, cs, infos = [], [], [], []

    for line in batch:
        x, y, c, i = line

        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)

        if y is not None:
            y = torch.tensor(np.asarray(y))

        if c is not None:
            c = torch.tensor(np.asarray(c))

        xs.append(x)
        ys.append(y)
        cs.append(c)
        infos.append(i)

    return xs, ys, cs, infos


def main(task, batchcount, batchsize, n_worker, model_type, feat_dim, n_cls, lr, score_show_file, ck_dir, device, epoch, max_n_aug, in_train_feat_0_list,in_train_feat_1_list, in_valid_feat_0_list,in_valid_feat_1_list,feat_LocationBlockTable_dir,feat_dir):
    train_ds = FeatDataset(in_train_feat_0_list,in_train_feat_1_list, cls_blance=True, batch_count=None, enchance=True,in_patch_ref_dir = feat_LocationBlockTable_dir,feat_dir = feat_dir)
    train_eval_ds = FeatDataset(in_train_feat_0_list,in_train_feat_1_list,cls_blance=True, batch_count=None, enchance=False,in_patch_ref_dir = feat_LocationBlockTable_dir,feat_dir = feat_dir )
    valid_ds = FeatDataset(in_valid_feat_0_list,in_valid_feat_1_list,cls_blance=False, batch_count=None, enchance=False,in_patch_ref_dir = feat_LocationBlockTable_dir,feat_dir = feat_dir)

    train_dl = torch.utils.data.DataLoader(
        train_ds,
        batch_size=batchsize,
        shuffle=True,
        drop_last=False,
        collate_fn=my_collate_func,
        num_workers=n_worker,
        persistent_workers=True and n_worker > 0,
    )

    train_eval_dl = torch.utils.data.DataLoader(
        train_eval_ds,
        batch_size=batchsize,
        shuffle=False,
        drop_last=False,
        collate_fn=my_collate_func,
        num_workers=n_worker,
        persistent_workers=False,
    )

    valid_dl = torch.utils.data.DataLoader(
        valid_ds,
        batch_size=batchsize,
        shuffle=False,
        drop_last=False,
        collate_fn=my_collate_func,
        num_workers=n_worker,
        persistent_workers=False,
    )

    report_out_file = f'{ck_dir}/valid_last_report.txt'
    csv_pred_out_file = f'{ck_dir}/valid_last_pred_out.csv'

    # ----------------------------------------------------------------------------
    # 选择模型
    if model_type == 'rtransformer':
        from rtransformer.model_rtransformer import Net
        from rtransformer.rtransformer_train_data import TrainData

        net = Net(feat_dim, n_cls, inter_dim=512, group_size=256, group_batch_size=512)
        traindata = TrainData(task, net, lr, n_cls, score_show_file, report_out_file, csv_pred_out_file)

    elif model_type == 'rtransformer_v2':
        from rtransformer.model_rtransformer_v2 import Net
        from rtransformer.rtransformer_train_data import TrainData

        net = Net(feat_dim, n_cls, inter_dim=512, n_block=6, expand_dim=768, squeeze_dim=256, bucket_size=256, bucket_batch_num=512)
        traindata = TrainData(task, net, lr, n_cls, score_show_file, report_out_file, csv_pred_out_file)


    elif model_type == 'clam_sb':
        from clam.clam_train_data import TrainData

        clam_kwargs = dict(is_sb=False, inst_loss_func='svm', feat_dim=feat_dim, n_classes=n_cls, size_arg='small', dropout=True, k_sample=8, subtyping=False)
        traindata = TrainData(task, clam_kwargs, lr, n_cls, score_show_file, report_out_file, csv_pred_out_file)

    elif model_type == 'clam_mb':
        from clam.clam_train_data import TrainData

        clam_kwargs = dict(is_sb=False, inst_loss_func='svm', feat_dim=feat_dim, n_classes=n_cls, size_arg='small', dropout=True, k_sample=8, subtyping=True)
        traindata = TrainData(task, clam_kwargs, lr, n_cls, score_show_file, report_out_file, csv_pred_out_file)

    elif model_type == 'la_mil':
        from t_mil.model_tmil import T_MIL
        from t_mil.tmil_train_data import TrainData

        net = T_MIL(feat_dim=feat_dim, n_classes=n_cls, latent_dim=512, num_heads=8, architecture='LA_MIL', max_limit_patch=15000)
        traindata = TrainData(task, net, lr, n_cls, score_show_file, report_out_file, csv_pred_out_file)

    elif model_type == 'ga_mil':
        from t_mil.model_tmil import T_MIL
        from t_mil.tmil_train_data import TrainData

        net = T_MIL(feat_dim=feat_dim, n_classes=n_cls, latent_dim=512, num_heads=8, architecture='GA_MIL')
        traindata = TrainData(task, net, lr, n_cls, score_show_file, report_out_file, csv_pred_out_file)

    elif model_type == 'dtfd_mil':
        from dtfd_mil.model_dtfd import DTFD_MIL
        from dtfd_mil.dtfd_train_data import TrainData

        net = DTFD_MIL(feat_dim=feat_dim, n_cls=n_cls, mDim=512, numLayer_Res=8, distill='MaxMinS', numGroup=1, instance_per_group=1)
        traindata = TrainData(task, net, lr, n_cls, score_show_file, report_out_file, csv_pred_out_file)

    else:
        raise NotImplementedError

    # ----------------------------------------------------------------------------

    last_ck = f'{ck_dir}/last.ckpt'
    if not os.path.exists(last_ck):
        last_ck = None

    ckpt_cb_list = []

    last_ckpt_cb = ModelCheckpoint(dirpath=ck_dir,
                                   filename='weight-{epoch:03d}',
                                   monitor='epoch',
                                   mode="max",
                                   save_top_k=1,
                                   save_last=True,
                                   verbose=False,
                                   )
    ckpt_cb_list.append(last_ckpt_cb)

    if task == 'single_class':
        best_ckpt_cb = ModelCheckpoint(
            dirpath=ck_dir,
            filename='best-{epoch:03d}-{val_f1:.4f}',
            monitor='val_f1',
            mode='max',
            save_top_k=1,
            save_last=True,
        )
        best_ckpt_cb.CHECKPOINT_NAME_LAST = 'best'
        ckpt_cb_list.append(best_ckpt_cb)

        max_cls1_auc_ckpt_cb = ModelCheckpoint(
            dirpath=ck_dir,
            filename='max-cls1-auc-{epoch:03d}-{val_cls1_auc:.4f}',
            monitor='val_cls1_auc',
            mode='max',
            save_top_k=1,
            save_last=True,
        )
        max_cls1_auc_ckpt_cb.CHECKPOINT_NAME_LAST = 'max-cls1-auc'
        ckpt_cb_list.append(max_cls1_auc_ckpt_cb)

    trainer = pl.Trainer(
        accelerator='gpu',
        devices=[device],
        max_epochs=epoch,
        # limit_train_batches=batchcount,
        detect_anomaly=False,
        default_root_dir=ck_dir,
        callbacks=ckpt_cb_list
    )
    trainer.fit(traindata, train_dl, [train_eval_dl, valid_dl], None, ckpt_path=last_ck)


if __name__ == '__main__':
    from t1_cfg import *

    main(
        task=task,
        batchcount=batchcount,
        batchsize=batchsize,
        n_worker=n_worker,
        model_type=model_type,
        feat_dim=feat_dim,
        n_cls=n_cls,
        lr=lr,
        score_show_file=score_show_file,
        ck_dir=ck_dir,
        device=device,
        epoch=epoch,
        max_n_aug=max_n_aug,
        in_train_feat_0_list = in_train_feat_0_list,
        in_train_feat_1_list = in_train_feat_1_list,

        in_valid_feat_0_list = in_valid_feat_0_list,
        in_valid_feat_1_list = in_valid_feat_1_list,
        feat_LocationBlockTable_dir = feat_LocationBlockTable_dir,
        feat_dir = feat_dir
    )
