import torch
import torch.nn as nn

from attention import MultiHeadAttention
from embedding import PositionalEncoding
from encoder import TransformerBlock
from utilities import replikasi


class DecoderBlock(nn.Module):
    """
    Blok dekoder dalam arsitektur transfomer yang terdiri dari dua komponen
    - multihead self-attention untuk mengangkap hubungan antar token dalam input
    - transformerBlock untuk interaksi dengan output dari blok encoder

    Parameter:
        dimensi_embedding (int): dimensi vektor embedding (nilai default 512)
        heads (int): jumlah head dalam multihead attention (nilai default 8)
        faktor_ekspansi (int): faktor ekspansi untuk feed-forward network di transformer
        dropout (float): probabilitias dropout untuk regularisasi (nilai default 0.2)

    Attribut:
        attention (MultiHeadAttention): lapisan multi-head self-attention untuk input decoder
        norm (nn.LayerNorm): layer normalisasi untuk stabilitasi pelatihan
        dropout (nn.Dropout): Dropout untuk regularisasi
        transformerBlock (TransformerBlock): block transformer yang menggabungkan interaksi
                                            encoder decoder

    Proses forward:
    - self-attention pada input decoder
        - menggunakan `MultiHeadAttention` untuk menghitung perhatian antar token dalam input decoder
        - mask diterapkan untuk mencegah perhatian ke token future (look-ahead masking)
    - residual connection dan normalization
        - menambahkan hasil self-attention ke input asli sebagai residual connection
        - menerapkan layer normalization untuk stabilitasi pelatihan

    - feed-forward melalui transformerBlock
        - output dari langkah sebelumnya dilewatkan ke `TransformerBlock`
        - `TransformerBlock` melakukan interaksi antara decoder dan decoder menggunakan key
            dan query dari encoder

    Informasi tambahan:
        - menggunakan residual connection setelah self-attention untuk mempertahankan informasi
            input
        - layer normalization membantu menjaga distribusi aktivasi tetap stabil
        - dropout diterapkan setelah residual connection untuk regularisasi
        - kompatible dengan mask untuk skenario seperti padding atau look-ahead

        - `dimensi_embedding` harus dibagi dengan `heads` dalam `MultiHeadAttention`
        - `TransformerBlock` bertanggun jawab atas interaksi encoder-decoder
        - dropout digunakan untuk mencegah overfitting selama proses training

    Informasi input:
        - tensor `key`, `query`, `x` harus memiliki dimensi embedding yang sama
        - mask harus memiliki bentukl yang kompatible dengan (batch, 1, x_len, x_len)
    """

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
    """
    Implementasi lengkap dari komponen encoder dalam arsitek transformer
    decoder bertanggung jawab untuk menghasilkan output berdasarkan input target
    dan interaksi dengan output dari encoder

    Parameter:
        - ukuran_target_vocab (int): ukuran kosakata target (jumlah token unik dalam target)
        - panjang_sekuens (int): panjang maksimum sekuens yang dapat diproses oleh decoder
        - dimensi_embedding (int): dimensi vektor embedding (default nilainya 512)
        - jumlah_blocks (int): jumlah block decoder dalam stack (default nilainya 6)
        - faktor_ekspansi (int): faktor ekspansi untuk feed forward network di setiap blok decoder
        - heads (int): jumlah head dalam multi-head attention (nilainya default 8)
        - dropout (float): probabilitias dropout untuk regularisasi (nilainya default 8)

    Attribut:
        embedding (nn.Embedding): lapisan embedding untuk mengkonversi token target
        positional_encoder (PositionalEncoding): lapisan dari positional encoding untuk menambahkan informasi posisi
        dropout (nn.Dropout): dropout untuk regularisasi setelah embedding dan position embbeding

    Forward parameter:
        x (torch.Tensor): input tensor target dengan bentuk (batch, seq_len)
        encoder_output (torch.Tensor): output tensor dari encoder dengan bentuk
                                        (batch, seq_len_encoder, dimensi_embedding)
        mask (torch.Tensor): mask untuk self-attention dengan bentuk (batch, 1, seq_len, seq_len)

    Forward return:
        torch.Tensor: output tensor dengan bentuk (batch, seq_len, dimensi_embedding)

    Proses:
        - mengkonversi token targetnya menjadi vektor embeding menggunakan `nn.Embedding`
        - menambahkan informasi posisi vektor embedding menggunakan `PositionalEncoding`
        - proses diatas penting untuk memberikan konteks urutan nantinya kepada model
        - menerapkan dropout pada hasil embedding dan positional encoding untuk regularisasi
        - input dilewatkan melalui stack block decoder
        - setiap block melakukan self-attention pada input decoder dan interaksi dengan
            output encoder
        - output nantinya dari blok dekoder terakhhir dekembalikan sebagai hasil akhir

    Informasi tambahan:
        - menggunakan positional encoding untuk menbahkan informasi posisi ke input
        - stack block decoder memungkinkan pemodelan hubungan kompleks antar-token
        - ini kompatible dengan mask untuk skenario seperti padding atau look-ahead
    """
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
