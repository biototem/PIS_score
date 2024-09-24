import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.utils.data
from my_py_lib import class_eval_tool
from my_py_lib import numpy_tool
from my_py_lib.path_tool import open2
from sklearn.metrics import confusion_matrix, classification_report, f1_score
from sklearn.metrics import roc_curve, auc, precision_recall_fscore_support
from scipy.stats import spearmanr
from timm.scheduler import CosineLRScheduler
from sp_reg_loss_mse_ordinal_loss import SpRegLossMseOrdinalLoss_R03_Z200, SpRegLossOrdinalLoss_R03_Z100


def to_str_list(ns):
    # assert isinstance(ns, np.ndarray) and ns.ndim == 1
    ns = [str(n) for n in ns]
    return ns


def calc_each_class_auc(batch_labels, batch_probs, labels):
    out_auc_scores = []
    for cls in labels:
        bools = batch_labels == cls
        fpr, tpr, _ = roc_curve(bools, batch_probs[:, cls])
        auc_score = auc(fpr, tpr)
        if np.isnan(auc_score):
            print('Warning! Found auc is Nan.')
            auc_score = 0.
        out_auc_scores.append(auc_score)
    return out_auc_scores


def make_task_train_y(task, device, n_cls, labels):
    # 制作能用来训练的 目标 y
    if task == 'single_class':
        y = torch.stack(labels, 0)
        y = y.type(torch.long)

    elif task == 'multi_binary_class':
        y = torch.zeros([len(labels), n_cls], dtype=torch.float32, device=device)
        for i in range(len(labels)):
            y[i, labels[i]] = 1.

    elif task in ('regression', 'regression-mse-ord-r03-z200', 'regression-ord-r03-z100'):
        y = torch.stack(labels, 0)

    else:
        raise AssertionError(f'Error! Bad task {task}')

    return y


def task_loss_func(task, loss_func, n_cls, model_logit, labels):
    # 根据任务计算损失
    y = make_task_train_y(task, model_logit.device, n_cls, labels)

    if task == 'single_class':
        loss = loss_func(model_logit, y)

    elif task == 'multi_binary_class':
        loss = loss_func(model_logit, y)

    elif task in ('regression', 'regression-mse-ord-r03-z200', 'regression-ord-r03-z100'):
        loss = loss_func(model_logit, y)

    else:
        raise AssertionError(f'Error! Bad task {task}')

    return loss


class TrainData(pl.LightningModule):
    def __init__(self, task, lr=2e-4, n_cls=6, score_show_file=None, report_out_file=None, csv_pred_out_file=None, is_test=False):
        super().__init__()
        # 基构造函数，只处理杂事

        self.task = task
        self.n_cls = n_cls

        if task == 'single_class':
            self.loss_func = nn.CrossEntropyLoss(label_smoothing=0.2)
        elif task == 'multi_binary_class':
            self.loss_func = nn.BCEWithLogitsLoss()
        elif task == 'regression':
            self.loss_func = nn.SmoothL1Loss()
        elif task == 'regression-mse-ord-r03-z200':
            self.loss_func = SpRegLossMseOrdinalLoss_R03_Z200()
        elif task == 'regression-ord-r03-z100':
            self.loss_func = SpRegLossOrdinalLoss_R03_Z100()
        else:
            raise AssertionError(f'Error! Invalid task param {task}')

        self.lr = lr

        self.score_show_file = score_show_file
        self.score_history = []

        self.report_out_file = report_out_file
        self.csv_pred_out_file = csv_pred_out_file

        self._ignored_first_temp_score_line = False
        self.is_test = is_test

        # 临时缓存，注意该缓存为 valid 和 test 共用，但结构不一样
        self._metric_cache = {}

    def forward(self, x):
        # 需要手动实现
        raise NotImplementedError()

    def forward_for_extract_heat(self, x):
        # 需要手动实现
        raise NotImplementedError()

    def training_step(self, batch, batch_idx):
        # 需要手动实现
        # 下面屏蔽的是参考代码
        raise NotImplementedError()
        # x, y, c, infos = batch
        # out = self.forward(x)
        #
        # loss = task_loss_func(self.task, self.loss_func, self.n_cls, out, y)
        #
        # self.log("train_loss", loss, prog_bar=True, rank_zero_only=True)
        # return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        # 需要手动实现
        # 下面屏蔽的是参考代码
        raise NotImplementedError()
        # x, y, c, infos = batch
        #
        # out = self.forward(x)
        #
        # loss = task_loss_func(self.task, self.loss_func, self.n_cls, out, y)
        #
        # out = out.cpu().numpy()
        # y = [each_y.cpu().numpy() for each_y in y]
        # loss = loss.item()
        #
        # cur_metric_cache = self._metric_cache.setdefault(dataloader_idx, [])
        # cur_metric_cache.append([out, y, loss, infos])

    def on_validation_epoch_end(self):

        csv_table_outs = [['dl_i', 'relpath', 'label_cls', 'pred_cls', 'pred_prob...']]

        for dl_i, lines in self._metric_cache.items():
            if dl_i == 0 and not self.is_test:
                data_name = 'train'
            elif dl_i == 1 and not self.is_test:
                data_name = 'val'
            else:
                data_name = f'dl_{dl_i}'

            outs = []
            ys = []
            losses = []
            infos = []

            for line in lines:
                outs.extend(line[0])
                ys.extend(line[1])
                losses.append(line[2])
                infos.extend(line[3])

            outs = np.stack(outs, 0)
            # ys = np.concatenate(ys, 0)
            losses = np.stack(losses, 0)

            eval_loss = np.float32(losses).mean()

            report = f'{data_name} epoch={self.current_epoch} loss={eval_loss}'

            if self.task == 'single_class':
                ys = np.stack(ys, 0)

                pred_cls = np.argmax(outs, 1)
                probs = numpy_tool.softmax(outs, 1)

                eval_f1 = f1_score(ys, pred_cls, average='macro', zero_division=0)  #macro  -----binary
                report += f' f1={eval_f1}'

                eval_each_auc = calc_each_class_auc(ys, probs, range(self.n_cls))
                eval_auc = np.mean(eval_each_auc)
                report += f' auc={eval_auc}'
                for cls in range(self.n_cls):
                    report += f' cls{cls}_auc={eval_each_auc[cls]}'

                eval_cm = str(confusion_matrix(ys, pred_cls, labels=range(self.n_cls)))
                report += f'\n\n{eval_cm}'

                eval_cls_report = classification_report(ys, pred_cls, digits=3, zero_division=0, labels=range(self.n_cls))
                report += f'\n\n{eval_cls_report}'


                eval_each_prec, eval_each_recall, eval_each_f1, eval_each_support = precision_recall_fscore_support(ys, pred_cls, labels=range(self.n_cls),
                                                                                                                    zero_division=0)

                _, _, eval_each_f2, _ = precision_recall_fscore_support(ys, pred_cls, beta=2, labels=range(self.n_cls), zero_division=0)
                _, _, eval_each_f05, _ = precision_recall_fscore_support(ys, pred_cls, beta=0.5, labels=range(self.n_cls), zero_division=0)

                for cls in range(self.n_cls):
                    self.log(f'{data_name}_cls{cls}_auc', eval_each_auc[cls], add_dataloader_idx=False)
                    self.log(f'{data_name}_cls{cls}_prec', eval_each_prec[cls], add_dataloader_idx=False)
                    self.log(f'{data_name}_cls{cls}_recall', eval_each_recall[cls], add_dataloader_idx=False)
                    self.log(f'{data_name}_cls{cls}_f1', eval_each_f1[cls], add_dataloader_idx=False)
                    self.log(f'{data_name}_cls{cls}_f2', eval_each_f2[cls], add_dataloader_idx=False)
                    self.log(f'{data_name}_cls{cls}_f05', eval_each_f05[cls], add_dataloader_idx=False)

                self.log(f'{data_name}_f1', eval_f1, add_dataloader_idx=False)
                self.log(f'{data_name}_auc', eval_auc, add_dataloader_idx=False)



            elif self.task == 'multi_binary_class':
                probs = numpy_tool.sigmoid(outs)
                pred_cls_bools = probs > 0.5
                pred_cls = [np.argwhere(b).reshape(-1) for b in pred_cls_bools]

                assert len(ys) == len(pred_cls)

                tables = []
                for a1, a2 in zip(pred_cls, ys):
                    t = class_eval_tool.calc_class_score(a1, a2, range(self.n_cls))
                    tables.append(t)

                score_table = class_eval_tool.accumulate_class_score(tables, range(self.n_cls))

                avg_f05 = []
                avg_f1 = []
                avg_f2 = []
                for cls, table in score_table.items():
                    avg_f05.append(table['f05'])
                    avg_f1.append(table['f1'])
                    avg_f2.append(table['f2'])
                    self.log(f'{data_name}_cls{cls}_f05', table['f05'], add_dataloader_idx=False)
                    self.log(f'{data_name}_cls{cls}_f1', table['f1'], add_dataloader_idx=False)
                    self.log(f'{data_name}_cls{cls}_f2', table['f2'], add_dataloader_idx=False)
                avg_f05 = float(np.mean(avg_f05))
                avg_f1 = float(np.mean(avg_f1))
                avg_f2 = float(np.mean(avg_f2))
                self.log(f'{data_name}_f05', avg_f05, add_dataloader_idx=False)
                self.log(f'{data_name}_f1', avg_f1, add_dataloader_idx=False)
                self.log(f'{data_name}_f2', avg_f2, add_dataloader_idx=False)
                report += f' f05={avg_f05:.5f} f1={avg_f1:.5f} f2={avg_f2:.5f}'

                # 添加数据到 csv 表
                for each_info, each_y, each_pred_cls, each_pred_probs in zip(infos, ys, pred_cls, probs):
                    rel_meta_file = each_info['rel_meta_file']
                    csv_line = [data_name, rel_meta_file, ':'.join(to_str_list(each_y)), ':'.join(to_str_list(each_pred_cls)), *to_str_list(each_pred_probs)]
                    csv_table_outs.append(csv_line)
                #

            elif self.task in ('regression', 'regression-mse-ord-r03-z200', 'regression-ord-r03-z100'):
                ys = np.stack(ys, 0)
                avg_l1 = np.abs(outs - ys).mean()
                report += f' L1={avg_l1}'

                self.log(f'{data_name}_L1', avg_l1, add_dataloader_idx=False)

                # 增加相关系数计算
                # 线性相关系数
                person_corrs = []
                # 排序相关系数
                spearman_corrs = []
                for cls in range(self.n_cls):
                    p_corr = np.corrcoef(ys[:, cls], outs[:, cls])[0, 1]
                    s_corr = spearmanr(ys[:, cls], outs[:, cls])[0]
                    person_corrs.append(p_corr)
                    spearman_corrs.append(s_corr)
                    self.log(f'{data_name}_cls{cls}_person_corr', p_corr, add_dataloader_idx=False)
                    self.log(f'{data_name}_cls{cls}_spearman_corr', s_corr, add_dataloader_idx=False)
                    report += f' cls{cls}_person_corr={p_corr} cls{cls}_spearman_corr={s_corr}'

                avg_person_corr = np.mean(person_corrs)
                avg_spearman_corr = np.mean(spearman_corrs)
                self.log(f'{data_name}_person_corr', avg_person_corr, add_dataloader_idx=False)
                self.log(f'{data_name}_spearman_corr', avg_spearman_corr, add_dataloader_idx=False)
                report += f' person_corr={avg_person_corr} spearman_corr={avg_spearman_corr}'

                # 添加数据到 csv 表
                for each_info, each_y, each_pred in zip(infos, ys, outs):
                    rel_meta_file = each_info['rel_meta_file']
                    csv_line = [data_name, rel_meta_file, ':'.join(to_str_list(each_y)), ':'.join(to_str_list(each_pred)), *to_str_list(each_pred)]
                    csv_table_outs.append(csv_line)
                #

            else:
                raise AssertionError(f'Error! Invalid task param {self.task}')

            self.log(f'{data_name}_loss', eval_loss, add_dataloader_idx=False)

            report += f'\n'
            self.score_history.append(report)

        # 忽略运行后第一行记录，该行是快速测试生成的
        if not self._ignored_first_temp_score_line and not self.is_test:
            self._ignored_first_temp_score_line = True
            return

        if self.score_show_file is not None:
            open2(self.score_show_file, 'w', encoding='utf8').write(''.join(self.score_history))

        if self.report_out_file is not None:
            open2(self.report_out_file, 'w').write(report)

        if self.csv_pred_out_file is not None:
            text = []
            for l_1 in csv_table_outs:
                l_2 = ','.join([str(i) for i in l_1])
                text.append(l_2)
            text = '\n'.join(text)
            open2(self.csv_pred_out_file, 'w').write(text)

        self._metric_cache.clear()

    def test_step(self, *args, **kwargs):
        self.validation_step(*args, **kwargs)

    def on_test_epoch_end(self, *args, **kwargs):
        self.on_validation_epoch_end(*args, **kwargs)

    def configure_optimizers(self):
        optim = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=4e-4)
        scheduler = CosineLRScheduler(optim, t_initial=2, lr_min=1e-7,
                                      cycle_mul=2, cycle_decay=1, cycle_limit=20,
                                      warmup_t=6, warmup_lr_init=1e-6, warmup_prefix=True)
        return [optim], [{"scheduler": scheduler, "interval": "epoch"}]

    def lr_scheduler_step(
            self,
            scheduler,
            metric,
    ) -> None:
        scheduler.step(epoch=self.current_epoch)

    def on_save_checkpoint(self, checkpoint):
        checkpoint['score_history'] = self.score_history

    def on_load_checkpoint(self, checkpoint):
        self.score_history = checkpoint['score_history']
