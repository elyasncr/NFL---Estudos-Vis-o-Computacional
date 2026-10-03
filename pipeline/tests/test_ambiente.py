import pytest


@pytest.mark.model
def test_gpu_blackwell_disponivel():
    import torch

    assert torch.cuda.is_available(), "PyTorch não enxerga a GPU; confira o índice cu128"
    assert torch.cuda.get_device_capability(0) >= (12, 0)
    x = torch.ones(1024, device="cuda")
    assert float((x * 2).sum()) == 2048.0
