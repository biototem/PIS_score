'''
mod from https://zhuanlan.zhihu.com/p/482153702
'''

import torch
import torch.nn as nn


class OrdinalRegressionLoss(nn.Module):

    def __init__(self, n_cls, train_cutpoints=False, scale=20.0, eps=1e-15):
        super().__init__()
        self.n_cls = n_cls
        self.eps = eps
        n_cutpoints = self.n_cls - 1
        self.cutpoints = nn.Parameter(torch.arange(n_cutpoints, dtype=torch.float32) * scale / (n_cls - 2) - scale / 2)
        if not train_cutpoints:
            self.cutpoints.requires_grad_(False)

    def forward(self, pred, label):
        sigmoids = torch.sigmoid(self.cutpoints - pred[...,None])
        link_mat = sigmoids[..., 1:] - sigmoids[..., :-1]
        link_mat = torch.cat([
            sigmoids[..., :1],
            link_mat,
            1 - sigmoids[..., -1:]
        ],
            dim=-1
        )

        eps = self.eps
        likelihoods = torch.clamp(link_mat, eps, 1 - eps)

        neg_log_likelihood = torch.log(likelihoods)
        if label is None:
            loss = 0
        else:
            loss = -torch.gather(neg_log_likelihood, -1, label[..., None]).mean()

        return loss, likelihoods


class OrdinalRegressionLoss2(nn.Module):

    def __init__(self, n_cls, train_cutpoints=True, init_cutpoints_range=(0, 3), eps=1e-8):
        super().__init__()
        self.n_cls = n_cls
        self.eps = eps
        n_cutpoints = self.n_cls - 1
        self.cutpoints = nn.Parameter(torch.linspace(init_cutpoints_range[0], init_cutpoints_range[1], steps=n_cutpoints, dtype=torch.float32))
        if not train_cutpoints:
            self.cutpoints.requires_grad_(False)

    def forward(self, pred, label):
        sigmoids = torch.sigmoid(self.cutpoints - pred[...,None])
        link_mat = sigmoids[..., 1:] - sigmoids[..., :-1]
        link_mat = torch.cat([
            sigmoids[..., :1],
            link_mat,
            1 - sigmoids[..., -1:]
        ],
            dim=-1
        )

        eps = self.eps
        likelihoods = torch.clamp(link_mat, eps, 1 - eps)

        neg_log_likelihood = torch.log(likelihoods)
        loss = -torch.gather(neg_log_likelihood, -1, label[..., None]).mean()

        return loss #, likelihoods


if __name__ == '__main__':
    # 使用样例
    ord_loss = OrdinalRegressionLoss(6, train_cutpoints=True)
    pred = torch.rand((6, 2), requires_grad=True)
    label = torch.randperm(len(pred)).reshape(-1, 1).repeat(1, 2)

    print(label.tolist())

    optim = torch.optim.Adam([pred] + list(ord_loss.parameters()), 1e-4)

    for it in range(2 ** 32):
        optim.zero_grad()
        loss, likelihoods = ord_loss(pred, label)
        loss += (pred - label).abs().mean()
        loss.backward()
        optim.step()
        if it % 1000 == 0:
            # print(ord_loss.cutpoints.grad.tolist())
            print(loss.item(), pred.squeeze().tolist())
