import pytest
import torch

@pytest.fixture(autouse=True)
def _set_seed():
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
