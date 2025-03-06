import pytest
import torch.nn as nn
from torch import isin
from torch._prims_common import is_integer_dtype

from utilities import replikasi


class BlockDummy(nn.Module):
    def __init__(self, input_size: int, output_size: int) -> None:
        super(BlockDummy, self).__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x) -> nn.Linear:
        return self.linear(x)


def test_replikasi_default() -> None:
    """
    test fungsi replikasi dengan parameter default
    """
    block = BlockDummy(10, 5)
    blok_replikasi = replikasi(block)
    assert isinstance(blok_replikasi, nn.ModuleList), (
        "Output harus berupa nn.ModuleList"
    )
    assert len(blok_replikasi) == 6, "Jumlah blok harus sama dengan nilai default N = 6"

    for replikasi_block in blok_replikasi:
        assert isinstance(replikasi_block, BlockDummy), (
            "setiap elemen harus instance BlockDummy"
        )
        assert replikasi_block is not block, (
            "setiap block harus deep copy, bukan referensi asli"
        )


def test_replikasi_custom_N() -> None:
    block = BlockDummy(8, 4)
    N: int = 3
    replikasi_blocks = replikasi(block, N)
    assert len(replikasi_blocks) == N, f"jumlah blok harus sama dengan nilai N={N}"

    for replikasi_block in replikasi_blocks:
        assert isinstance(replikasi_block, BlockDummy), (
            "setiap elemen harus instance BlockDummy"
        )
        assert replikasi_block is not block, (
            "setiap block harus deep copy, bukan referensi asli"
        )
