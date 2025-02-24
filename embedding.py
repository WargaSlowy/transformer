from math import sin, cos, sqrt, log
import torch
import torch.nn as nn

class Embedding(nn.Module):
    def __init__(self, ukuran_vocab: int, dimensi_embedding: int) -> None:
        super(Embedding, self).__init__()
        self.dimensi_embedding = dimensi_embedding
        self.embed = nn.Embedding(num_embeddings=ukuran_vocab, embedding_dim=dimensi_embedding)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.embed(x) * sqrt(self.dimensi_embedding)
        return output

class PositionalEncoding(nn.Module):
    def __init__(self, dimensi_embedding: int, panjang_maksimal_sekuens: int = 5_000, dropout: float = 0.1) -> None:
        super(PositionalEncoding, self).__init__()
        self.dimensi_embedding = dimensi_embedding
        self.dropout = nn.Dropout(p=dropout)

        positional_encoding = torch.zeros(panjang_maksimal_sekuens, self.dimensi_embedding)
        position = torch.arange(0, panjang_maksimal_sekuens).unsqueeze(1)

        div_term = torch.exp(torch.arange(0, dimensi_embedding, 2) * (log(10_000.0) / dimensi_embedding))

        positional_encoding[:, 0::2] = torch.sin(position * div_term)
        positional_encoding[:, 1::2] = torch.cos(position * div_term)
        
        pe = positional_encoding.unsqueeze(0)
        self.register_buffer('pe', pe)

    def pe_sin(self, position: int, i: int) -> float:
        return sin(position / (10_000 ** ((2 * i) / self.dimensi_embedding)))

    def pe_cos(self, position: int, i: int) -> float:
        return cos(position / (10_000 ** ((2 * i) / self.dimensi_embedding)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1)].requires_grad_(False)
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
    
    embedding_layer = Embedding(ukuran_vocab=ukuran_vocab, dimensi_embedding=dimensi_embedding)
    positional_encoding_layer = PositionalEncoding(dimensi_embedding=dimensi_embedding, panjang_maksimal_sekuens=panjang_maksimal_sekuens)

    embed_output = embedding_layer(input_sekuens)
    print("Output dari embedding (representasi numerik): ")
    print(embed_output.shape)

    output_dengan_positional_encoding = positional_encoding_layer(embed_output)
    print("output dengan positional encoding: ")
    print(output_dengan_positional_encoding.shape)

    print("nilai output dengan positional encoding:")
    print(output_dengan_positional_encoding[0, :2, :5])
