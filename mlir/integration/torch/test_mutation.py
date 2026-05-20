import torch
import torch.nn as nn

from integration.torch.check import check_backend


def test_cat():
    class CatNet(nn.Module):
        def forward(self, x: torch.Tensor):
            return torch.cat((x, x, x), 0)

    check_backend(CatNet().eval(), torch.randn(2, 3))

def test_transpose():
    class TransposeNet(nn.Module):
        def forward(self, x: torch.Tensor):
            return torch.transpose(x, 0, 1)

    check_backend(TransposeNet().eval(), torch.randn(2, 3))
