import torch
import torchvision.models as models

from integration.torch.check import check_backend, check_compile


def setup():
    """Return (eval-mode model, example_input) for ResNet-18 with random weights."""
    model = models.resnet18(weights=None)
    model.eval()
    x = torch.randn(1, 3, 224, 224)
    return model, x


def test_resnet18_compile():
    """docc backend output matches PyTorch eager output (default compile path)."""
    model, x = setup()
    check_compile(model, x, rtol=1e-3, atol=1e-4)


def test_resnet18_backend():
    """docc backend with target='none' output matches PyTorch eager output."""
    model, x = setup()
    check_backend(model, x, rtol=1e-3, atol=1e-4)
