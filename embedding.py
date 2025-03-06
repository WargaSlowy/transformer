from math import cos, log, sin, sqrt

import torch
import torch.nn as nn


class Embedding(nn.Module):
    """
    Kelas menghasilkan representasi numerik dari indeks kata dalam vocab

    Attribut:
        dimensi_embedding (int): dimensi embedding yang digunakan untuk representasi
                                setiap dari kata
        embed (nn.Embedding): embedding PyTorch yang mengonversi indeks kata
                                menjadi vektor embedding

    Metode:
        forward(x: torch.Tensor) -> torch.Tensor:
            hitung representasi embedding dari input tensor dengan skala
            akar kuadrat dari dimensi embedding
    """

    def __init__(self, ukuran_vocab: int, dimensi_embedding: int) -> None:
        """
        Inisialisasi lapisan embedding

        Parameter:
            ukuran_vocab (int): ukuran vocab (jumlah kata unik dalam dataset)
            dimensi_embedding (int): dimensi embedding untuk setiap dari kata
        """
        super(Embedding, self).__init__()
        self.dimensi_embedding = dimensi_embedding
        self.embed = nn.Embedding(
            num_embeddings=ukuran_vocab, embedding_dim=dimensi_embedding
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Fungsi untuk melakukan forward pass untuk result representasi embedding dari input
        tensor

        Parameter:
            x (torch.Tensor): tensor yang berisi indeks kata dengan
                                shape (batch_size, sequence_length)

        Return:
            torch.Tensor: representasi embedding dari input tensor dengan shape
                            (batch_size, sequence_length, dimensi_embedding)
        """
        output = self.embed(x) * sqrt(self.dimensi_embedding)
        return output


class PositionalEncoding(nn.Module):
    """
    Kelas untuk menambahkan positional encoding ke representasi embedding
    positional encoding akan membantu model akan memahami urutan dari token dalam sekuens

    Attribut:
        dimensi_embedding (int): dimensi embedding yang digunakan
        dropout (nn.Dropout): lapisan dropout untuk regularisasi
        pe (torch.Tensor): buffer yang menyimpan positional encoding untuk semua posisi

    Return:
        pe_sin(position: int, i: int) -> float: menghitung nilai sinus untuk positional
                                            encoding pada posisi tertentu
        pe_cos(position: int, i: int) -> float: menghitung cosinus positional encoding
                                            pada posisi tertentu
        forward(x: torch.Tensor) -> torch.Tensor: tambahkan positional encoding ke input
                                            tenso dan menerapkan dropout
    """

    def __init__(
        self,
        dimensi_embedding: int,
        panjang_maksimal_sekuens: int = 5_000,
        dropout: float = 0.1,
    ) -> None:
        """
        Inisialisasi lapisan dari positional encoding
        """
        super(PositionalEncoding, self).__init__()
        self.dimensi_embedding = dimensi_embedding
        self.dropout = nn.Dropout(p=dropout)

        positional_encoding = torch.zeros(
            panjang_maksimal_sekuens, self.dimensi_embedding
        )
        position = torch.arange(0, panjang_maksimal_sekuens).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, dimensi_embedding, 2) * (log(10_000.0) / dimensi_embedding)
        )

        positional_encoding[:, 0::2] = torch.sin(position * div_term)
        positional_encoding[:, 1::2] = torch.cos(position * div_term)

        pe = positional_encoding.unsqueeze(0)
        self.register_buffer("pe", pe)

    def pe_sin(self, position: int, i: int) -> float:
        """
        Menghitung komponen sinus dari positionla encoding untuk posisi dan dimensi
        tertentu

        Parameter:
            position (int): posisi token dalam urutan (dimulai dari 0)
            i (int): indeks dimensi dalam vektor embedding (dimulai dari 0)

        Return:
            (float): nilai sinus hasil perhitungan positional encoding

        Rumus:
            sin (position / (10_000 ** (2i / d_model)))
            - d_model = self.dimensi_embedding (dimensi vektor embedding)
            - 10_000 = konstanta dasar untuk skala eksponensial
            - 2i = memastikan pola periodik yang berbeda untuk setiap dari dimensi

        Informasi tambahan:
            - implementasi ini sesuai dari paper "attention is all you need"
                (Vaswani et al., 2017)
            - untuk dimensi genap, menggunakan sin; untuk ganjil maka gunakan cos
        """
        return sin(position / (10_000 ** ((2 * i) / self.dimensi_embedding)))

    def pe_cos(self, position: int, i: int) -> float:
        """
        Mnghitung komponen kosinus dari positional encoding untuk posisi dan dimensi tertentu

        Parameter:
            position (int): posisi token dalam urutan (dimulai dari 0)
            i (int): indeks dimensi dalam vektor embedding (dimulai dari 0)

        Return:
            float: nilai kosinus hasil dari perhitungan positional encoding

        Rumus:
            cos (position / (10_000 ** (2i / d_model)))
            - d_model = self.dimensi_embedding (dimensi vektor embedding)
            - 10_000 = kostanta dasar untuk skala eksponensial
            - 2i = memastikan pola periodik yang berbeda untuk setiap dimensi

        Informasi tambahan:
            - merupakan pasangan dari pe_sin() untuk membetuk positional encoding lengkap
            - nilai cos ini diambil untuk dimensi ganjil, sin untuk genap dalam implementasi
              standar
        """
        return cos(position / (10_000 ** ((2 * i) / self.dimensi_embedding)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Melakuan operasi forward pass dengan menambahkan positional encoding ke input

        Parameter:
            x (torch.Tensor): input tensor dengan bentuk (batch_size, sequence_length, dimensi_embedding)

        Return:
            torch.Tensor: tensor hasil penambahan positional encoding dengan dropout

        Proses:
            - mengambil positional encoding yang telah dihitung sebelumnya
            - melakukan slicing sesuai panjang sequence input
            - nonaktifkan gradient untuk positional encoding
            - menambahkan positional encoding ke input tensor
            - menerapkan dropout untuk regularisasi

        Informasi tambahan:
            - positional encoding bersifat fixed (tidak dipelajari selama training)
            - dropout diterapkan setelah penambahan positional encoding
            - menggunakan broadcasting untuk menyesuaikan panjang sequence
        """
        x = x + self.pe[:, : x.size(1)].requires_grad_(False)
        return self.dropout(x)


if __name__ == "__main__":
    ukuran_vocab: int = 1000
    dimensi_embedding: int = 512
    panjang_maksimal_sekuens: int = 20
    ukuran_batch: int = 2
    panjang_sekuens: int = 10

    input_sekuens = torch.randint(0, ukuran_vocab, (ukuran_batch, panjang_sekuens))
    print("input sekuens (indeks kata)")
    print(input_sekuens)

    embedding_layer = Embedding(
        ukuran_vocab=ukuran_vocab, dimensi_embedding=dimensi_embedding
    )
    positional_encoding_layer = PositionalEncoding(
        dimensi_embedding=dimensi_embedding,
        panjang_maksimal_sekuens=panjang_maksimal_sekuens,
    )

    embed_output = embedding_layer(input_sekuens)
    print("Output dari embedding (representasi numerik): ")
    print(embed_output.shape)

    output_dengan_positional_encoding = positional_encoding_layer(embed_output)
    print("output dengan positional encoding: ")
    print(output_dengan_positional_encoding.shape)

    print("nilai output dengan positional encoding:")
    print(output_dengan_positional_encoding[0, :2, :5])
