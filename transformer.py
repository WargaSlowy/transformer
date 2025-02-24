import torch
import torch.nn as nn
import torch.nn.functional as F

from decoder import Decoder
from encoder import Encoder


class Transformer(nn.Module):
    def __init__(
        self,
        dimensi_embedding: int,
        ukuran_sumber_vocab: int,
        ukuran_target_vocab: int,
        panjang_sekuens: int,
        jumlah_block: int = 6,
        faktor_ekspansi: int = 4,
        heads: int = 8,
        dropout: float = 0.2,
    ) -> None:
        super(Transformer, self).__init__()
        self.ukuran_target_vocab = ukuran_target_vocab

        self.encoder = Encoder(
            panjang_sekuens,
            ukuran_sumber_vocab,
            dimensi_embedding,
            jumlah_block,
            faktor_ekspansi,
            heads,
            dropout,
        )

        self.decoder = Decoder(
            ukuran_target_vocab,
            panjang_sekuens,
            dimensi_embedding,
            jumlah_block,
            faktor_ekspansi,
            heads,
            dropout,
        )

        self.fc_output = nn.Linear(dimensi_embedding, ukuran_target_vocab)

    def buat_mask_target(self, target: torch.Tensor) -> torch.Tensor:
        ukuran_batch, panjang_target = target.shape
        mask_target = torch.tril(torch.ones((panjang_target, panjang_target))).expand(
            ukuran_batch, 1, panjang_target, panjang_target
        )
        return mask_target

    def forward(self, sumber: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target_mask = self.buat_mask_target(target)
        encoder_output = self.encoder(sumber)
        output = self.decoder(target, encoder_output, target_mask)
        output = F.softmax(self.fc_output(outputs), dim=-1)
        return output


if __name__ == "__main__":
    ukuran_vocab_sumber: int = 11
    ukuran_vocab_target: int = 11
    jumlah_block: int = 6
    panjang_sekuens: int = 12

    sumber = torch.Tensor(
        [[0, 2, 5, 6, 4, 3, 9, 5, 2, 9, 10, 1], [0, 2, 8, 7, 3, 4, 5, 6, 7, 3, 10, 1]]
    )
    target = torch.Tensor(
        [[0, 1, 7, 4, 3, 5, 9, 2, 8, 10, 9, 1], [0, 1, 5, 6, 2, 4, 7, 6, 2, 8, 10, 1]]
    )
    print("sumber:")
    print(sumber.shape)
    print("target")
    print(target.shape)

    model = Transformer(
        512,
        ukuran_vocab_sumber,
        ukuran_vocab_target,
        panjang_sekuens,
        jumlah_block,
        faktor_ekspansi=4,
        heads=8,
    )

    print(model)
