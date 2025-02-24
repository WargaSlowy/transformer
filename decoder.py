import torch
import torch.nn as nn

from attention import MultiHeadAttention
from embedding import PositionalEncoding
from encoder import TransformerBlock
from utilities import replikasi


class DecoderBlock(nn.Module):
    def __init__(
        self,
        dimensi_embedding: int = 512,
        heads: int = 8,
        faktor_ekspansi: int = 4,
        dropout: float = 0.2,
    ) -> None:
        super(DecoderBlock, self).__init__()
        self.attention = MultiHeadAttention(dimensi_embedding, heads)
        self.norm = nn.LayerNorm(dimensi_embedding)
        self.dropout = nn.Dropout(dropout)
        self.transformerBlock = TransformerBlock(
            dimensi_embedding, heads, faktor_ekspansi, dropout
        )

    def forward(
        self,
        key: torch.Tensor,
        query: torch.Tensor,
        x: torch.Tensor,
        mask: torch.Tensor = None,
    ) -> torch.Tensor:
        decoder_attention = self.attention(x, x, x, mask)
        value = self.dropout(self.norm(decoder_attention + x))
        decoder_attention_output = self.transformerBlock(key, query, value)
        return decoder_attention_output


class Decoder(nn.Module):
    def __init__(
        self,
        ukuran_target_vocab: int,
        panjang_sekuens: int,
        dimensi_embedding: int = 512,
        jumlah_blocks: int = 6,
        faktor_ekspansi: int = 4,
        heads: int = 8,
        dropout: float = 0.2,
    ) -> None:
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(ukuran_target_vocab, dimensi_embedding)
        self.positionl_encoder = PositionalEncoding(dimensi_embedding, panjang_sekuens)
        self.blocks = replikasi(
            DecoderBlock(dimensi_embedding, heads, faktor_ekspansi, dropout)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor, encoder_output: torch.Tensor, mask: torch.Tensor = None
    ) -> torch.Tensor:
        x = self.dropout(self.positionl_encoder(self.embedding(x)))

        for block in self.blocks:
            x = block(encoder_output, x, encoder_output, mask)
        return x


if __name__ == "__main__":
    ukuran_target_vocab: int = 10_000
    panjang_sekuens: int = 50
    dimensi_embedding: int = 512
    jumlah_blocks: int = 6
    faktor_ekspansi: int = 4
    heads: int = 8
    dropout: float = 0.1

    decoder = Decoder(
        ukuran_target_vocab,
        panjang_sekuens,
        dimensi_embedding,
        jumlah_blocks,
        faktor_ekspansi,
        heads,
        dropout,
    )

    batch_size: int = 32
    x = torch.randint(0, ukuran_target_vocab, (batch_size, panjang_sekuens))
    encoder_output = torch.randn(batch_size, panjang_sekuens, dimensi_embedding)

    output = decoder(x, encoder_output)

    print(output.shape)
