import copy
import torch.nn as nn

def replikasi(block, N: int = 6):
    block_stack = nn.ModuleList([copy.deepcopy(block) for _ in range(N)])
    return block_stack

if __name__ == "__main__":
    class EncoderBlock(nn.Module):
        def __init__(self):
            super(EncoderBlock, self).__init__()
            self.layer = nn.Linear(10, 10)

        def forward(self, x):
            return self.layer(x)

    encoder_block = EncoderBlock()
    encoder_stack = replikasi(encoder_block, N=6)
    print(encoder_stack)
