"""
QUERY (Q), KEY (K), VALUE (V)

atensi(Q, K, V) = softmax(QK^T / SQRT(dK)) V
"""

from math import sqrt

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    def __init__(self, dimensi_embedding: int = 512, heads: int = 8) -> None:
        super(MultiHeadAttention, self).__init__()
        self.dimensi_embedding = dimensi_embedding
        self.heads = heads
        self.head = int(self.dimensi_embedding / self.heads)

        self.query = nn.Linear(self.head, self.head, bias=False)
        self.value = nn.Linear(self.head, self.head, bias=False)
        self.key = nn.Linear(self.head, self.head, bias=False)

        self.fc_output = nn.Linear(self.head * self.heads, dimensi_embedding)

    def forward(self, key: torch.Tensor, query: torch.Tensor, value: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        batch_size = key.size(0)
        k_len, q_len, v_len = key.size(1), query.size(1), value.size(1)

        key = key.reshape(batch_size, k_len, self.heads, self.head)
        query = query.reshape(batch_size, q_len, self.heads, self.head)
        value = value.reshape(batch_size, v_len, self.heads, self.head)

        key = self.key(key)
        query = self.query(query)
        value = self.value(value)

        product = torch.einsum("bqhd,bkhd->bhqk", [query, key])
        
        if mask is not None:
            product = product.masked_fill(mask == 0, float("-1e20"))

        product = product / sqrt(self.head)
        scores = F.softmax(product, dim=-1)

        output = torch.einsum("bhqv,bvhd->bqhd", [scores, value]).reshape(
            batch_size, q_len, self.heads * self.head
        )

        output = self.fc_output(output)
        return output

if __name__ == "__main__":
    dimensi_embedding: int = 512
    heads: int = 8
    attention_layer = MultiHeadAttention(dimensi_embedding, heads)

    batch_size: int = 32
    seq_len: int = 10
    key = torch.rand((batch_size, seq_len, dimensi_embedding))
    query = torch.rand((batch_size, seq_len, dimensi_embedding))
    value = torch.rand((batch_size, seq_len, dimensi_embedding))

    output = attention_layer(key, query, value)
    print(output.shape)
