import torch
import torch.nn as nn
import torch.nn.functional as F
from force_relative_import import enable_force_relative_import

with enable_force_relative_import():
    from .ordinal_reg_loss import OrdinalRegressionLoss2


class SpRegLossMseOrdinalLoss_R03_Z200(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse_loss_func = nn.MSELoss()

        self.ord_loss_range_min = 0
        self.ord_loss_range_max = 3
        self.ord_loss_range_seg = 200

        self.ordinal_reg_loss_func = OrdinalRegressionLoss2(self.ord_loss_range_seg, init_cutpoints_range=(self.ord_loss_range_min, self.ord_loss_range_max))

    def forward(self, input: torch.Tensor, target: torch.Tensor):
        target = target.type(torch.float32)

        loss1 = self.mse_loss_func(input, target)

        # 注意！必须减1e-4
        q_target = (target.clamp(self.ord_loss_range_min, self.ord_loss_range_max-1e-4) / self.ord_loss_range_max * self.ord_loss_range_seg).type(torch.long)
        loss2 = self.ordinal_reg_loss_func(input, q_target)

        return loss1 + loss2


class SpRegLossOrdinalLoss_R03_Z100(nn.Module):
    def __init__(self):
        super().__init__()
        self.ord_loss_range_min = 0
        self.ord_loss_range_max = 3
        self.ord_loss_range_seg = 100

        self.ordinal_reg_loss_func = OrdinalRegressionLoss2(self.ord_loss_range_seg, init_cutpoints_range=(self.ord_loss_range_min, self.ord_loss_range_max))

    def forward(self, input: torch.Tensor, target: torch.Tensor):
        target = target.type(torch.float32)

        # 注意！必须减1e-4
        q_target = (target.clamp(self.ord_loss_range_min, self.ord_loss_range_max-1e-4) / self.ord_loss_range_max * self.ord_loss_range_seg).type(torch.long)
        loss2 = self.ordinal_reg_loss_func(input, q_target)

        return loss2


if __name__ == '__main__':
    x = torch.rand([2, 2]) * 3
    y = torch.rand([2, 2]) * 3
    print(y.tolist())

    x.requires_grad_(True)

    optim = torch.optim.Adam([x], 1e-4)

    # loss_func = SpRegLossMseOrdinalLoss()
    loss_func = SpRegLossOrdinalLoss_R03_Z100()

    for _ in range(1000):
        loss = loss_func(x, y)
        optim.zero_grad()
        loss.backward()
        optim.step()
        print(x.tolist())
