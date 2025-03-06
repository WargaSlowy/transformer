import copy

import torch.nn as nn


def replikasi(block, N: int = 6) -> nn.ModuleList:
    """
    Membuat replikasi dari sebuah blok neural network sebanyak dari N kali

    Parameter:
        block (nn.Module): blok neural network yang akan direplikasi
                            harus merupakan instance dari `torch.nn.Module`
        N (int): jumlah replikasi yang diinginkan. default nilainya 6

    Return:
        nn.ModuleList: list dari blok neural network yang direplikasi
                        setiap elemen dalam list adalah salina mendalam dari
                        blok input
    """
    block_stack = nn.ModuleList([copy.deepcopy(block) for _ in range(N)])
    return block_stack


if __name__ == "__main__":

    class EncoderBlock(nn.Module):
        """
        Kelas representasi blok encoder dalam neural network

        Attribut:
            layer (nn.Linear): lapisan linear dengan input dan output ukuran 10
        """

        def __init__(self):
            super(EncoderBlock, self).__init__()
            self.layer = nn.Linear(10, 10)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """
            Fungsi forward pass untuk block decoder

            Parameter:
                x (torch.Tensor): input tensor dengan dimensi sesuai dengan
                                    lapisan linear
            Return:
                torch.Tensor: output tensor hasil transformasi linear
            """
            return self.layer(x)

    # membuat instance dari EncoderBlock
    encoder_block = EncoderBlock()
    # membuat replikasi dari EncoderBlock sebanyak 6 kali
    encoder_stack = replikasi(encoder_block, N=6)
    # testing hasil replikasi
    print(encoder_stack)
