from typing import Optional

import torch
import torch.nn as nn

from attention import MultiHeadAttention
from embedding import Embedding, PositionalEncoding


class TransformerBlock(nn.Module):
    """
    Blok dasar dari arsitektur Transformer yang menggabungkan mekanisme multi-head attention,
    feed-forward network, dan residual connection dengan layer normalization

    Attribut:
        attention (MultiHeadAttention): lapisan multi-head attention untuk mengangkap hubungan
                                        antar token
        norm (nn.LayerNorm): layer normaization untuk stabilitasi dalam proses pelatihan
        feed_forward (nn.Sequential): feed-forward network yang terdiri dari dua lapisan linear
                                     dengan aktivasi ReLU
        dropout (nn.Dropout): Blok dasar dari arsitektur Transformer yang menggabungkan
                             mekanisme multi-head attention, feed-forward network, dan
                            residual connection dengan layer normalization

    Forward parameter:
        key (torch.Tensor): tensor key dengan bentuk (batch, k_len, dimensi_embedding)
        query (torch.Tensor): tensor query dengan bentuk (batch, q_len, dimensi_embedding)
        value (torch.Tensor): tensor value dengan bentuk (batch, v_len, dimensi_embedding)
        mask (torch.Tensor): mask untuk attention mechanism dengan bentuk (batch, 1, q_len, k_len)

    Forward return:
        torch.Tensor: output tensor dengan bentuk (batch, q_len, dimensi_embedding)

    Proses:
        - menggunakan `MultiHeadAttention` untuk menghitung perhatian antar token berdasarkan key, query, dan value
        - mask diterapkan untuk mencegah perhatian ke token tertentu (misalnya padding atau look-ahead)
        - menambahkan hasil attention ke input value sebagi residual connection
        - menerapkan layer normalization untuk stabilitasi pelatihan
        - dropout diterapkan setelah normalization untuk regularisasi
        - melewatkan hasil dari langkah sebelumnya melalui feed-forward network
        - feed-forward network terdiri dari dua lapisan linear dengan aktivasi ReLU di antarnaya
        - menambahkan hasil feed-forward ke input sebelumnya sebagai residual connection
        - menerapkan layer normalization akhir

    Input data:
        - tensor key, query dan value harus memiliki dimensi embedding yang sama
        - mask harus memiliki bentuk yang kompatible dengan (batch, 1, q_len, k_len)
    """

    def __init__(
        self,
        dimensi_embedding: int = 512,
        heads: int = 8,
        faktor_ekspansi: int = 4,
        dropout: float = 0.2,
    ) -> None:
        super(TransformerBlock, self).__init__()

        self.attention = MultiHeadAttention(dimensi_embedding, heads)
        self.norm = nn.LayerNorm(dimensi_embedding)
        self.feed_forward = nn.Sequential(
            nn.Linear(dimensi_embedding, faktor_ekspansi * dimensi_embedding),
            nn.ReLU(),
            nn.Linear(faktor_ekspansi * dimensi_embedding, dimensi_embedding),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        key: torch.Tensor,
        query: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        output_attention = self.attention(key, query, value, mask)
        output_attention = output_attention + value
        normalisasi_attention = self.dropout(self.norm(output_attention))
        output_fc = self.feed_forward(normalisasi_attention)
        normalisasi_fc = self.dropout(self.norm(output_fc))
        return normalisasi_fc


class Encoder(nn.Module):
    def __init__(
        self,
        panjang_sekuens: int,
        ukuran_vocab: int,
        dimensi_embedding: int = 512,
        jumlah_block: int = 6,
        faktor_ekspansi: int = 4,
        heads: int = 8,
        dropout: float = 0.2,
    ) -> None:
        super(Encoder, self).__init__()
        self.embedding = Embedding(ukuran_vocab, dimensi_embedding)
        self.positional_encoder = PositionalEncoding(dimensi_embedding, panjang_sekuens)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(dimensi_embedding, heads, faktor_ekspansi, dropout)
                for _ in range(jumlah_block)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.positional_encoder(self.embedding(x))
        for block in self.blocks:
            output = block(output, output, output)
        return output


if __name__ == "__main__":
    ukuran_vocab: int = 10_000
    panjang_sekuens: int = 20
    ukuran_batch: int = 32
    dimensi_embedding: int = 512
    jumlah_block: int = 6
    heads: int = 8
    faktor_ekspansi: int = 4
    dropout: float = 0.2

    encoder = Encoder(
        panjang_sekuens,
        ukuran_vocab,
        dimensi_embedding,
        jumlah_block,
        faktor_ekspansi,
        heads,
        dropout,
    )
    input_tensor = torch.randint(0, ukuran_vocab, (ukuran_batch, panjang_sekuens))
    print(f"input tensor shape {input_tensor.shape}")

    output = encoder(input_tensor)
    print(f"output tensor shape: {output.shape}")
    ukuran_batch * panjang_sekuens * dimensi_embedding
