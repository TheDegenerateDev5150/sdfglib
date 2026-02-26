import torch
import torch.nn as nn

from torch_mlir import fx


class ConvNet(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, hidden_channels, kernel_size=3, padding=0, bias=False
        )
        self.conv2 = nn.Conv2d(
            hidden_channels, out_channels, kernel_size=3, padding=0, bias=False
        )

    def forward(self, x: torch.Tensor):
        h1 = self.conv1(x)
        h2 = self.conv2(h1)
        return h2


model = ConvNet(3, 16, 8)
example_input = torch.randn(1, 3, 32, 32)

torch_mlir = fx.export_and_import(
    model, example_input, output_type=fx.OutputType.LINALG_ON_TENSORS
)
print(str(torch_mlir))
