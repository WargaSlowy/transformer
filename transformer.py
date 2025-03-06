import torch
import torch.nn as nn
import torch.nn.functional as F

from decoder import Decoder
from encoder import Encoder


class Transformer(nn.Module):
    """
    Implementasi lengkap dari arsitektur Transformer yang terdiri dari encoder dan decoder
    model ini digunakan untuk tugas-tugas seperti machine translation, text summarization dll

    Parameter:
        dimensi_embedding (int): dimensi vektor embedding untuk representasi vektor
        ukuran_sumber_vocab (int): ukuran kosakata sumber (jumlah token unik dalam sumber vocab)
        ukuran_target_vocab (int): ukuran kosakata target (jumlah token unik dalam target vocab)
        panjang_sekuens (int): panjang sekuens yang dapat diproses oleh model
        jumlah_block (int): jumlah blok encoder dan decoder dalam stack (default nilai 6)
        faktor_ekspansi (int): faktor ekspansi untuk feed-forward network di setiap blok (default nilai 4)
        heads (int): jumlah head dalam multi-head attention (default nilai 8)
        dropout (float): probabilitas dropout untuk regularisasi (default nilai 0.27)

    Attribut:
        ukuran_target_vocab (int): menyimpan ukuran kosakata target
        encoder (Encoder): komponen encoder yang memproses input sumber
        decoder (Decoder): komponen decoder yang memproses input target dan interaksi dengan encoder
        fc_output (nn.Linear): lapisan linear untuk menghasilkan probabilitas token target

    Forward parameter:
        sumber (torch.Tensor): input tensor sumber dengan bentuk (batch, seq_len_sumber)
        target (torch.Tensor): input tensor target dengan bentuk (batch, seq_len_target)

    Forward return:
        torch.Tensor: output tensor dengan bentuk (batch, seq_len_target, ukuran_target_vocab),
                        berisi probabilitas untuk setiap token target

    Proses:
        - menggunakan `buat_mask_target` untuk membuat mask look-ahead untuk decoder
        - mask ini nantinya mencegah model melihat token masa depan selama decoding autoregresif
        - input sumber dilewatkan ke encoder untuk menghasilkan representasi kontekstual
        - input target dan output encoder dilewatkan ke decoder
        - mask target diterapkan untuk mencegah attention ke token masa depan
        - output decoder dilewatkan ke lapisan linear untuk menghasilkan logits
        - softmax diterapkan pada dimensi terakhir untuk menhasilkan distribusi probabilitas

    Informasi tambahan:
        - `Encoder` dan `Decoder` adalah kelas terpisah yang mengimplementasikan komponen utama
            transformer
        - dropout digunakan secara konsisten di encoder dan decoder untuk regularisasi

    Metode input:
        - tensor untuk sumber harus memiliki bentuk (batch, seq_len_sumber) dengan nilai indeks
        - tensor untuk terget harus memiliki bentuk (batch, seq_len_target) dengan nilai indeks
        - mask target dibuat secara otomatis oleh metode `buat_mask_target`
    """

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
