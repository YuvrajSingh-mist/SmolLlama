
from config import ModelArgs
import torch
import torch.nn as nn
import torch.nn.functional as F

class Normalization(nn.Module):
    def __init__(
        self,

        embeddings_dims: int = ModelArgs.embeddings_dims
    ):
        super().__init__()
        self.rmsnorm_layer = torch.nn.RMSNorm(normalized_shape=embeddings_dims)


    def forward(self, x):

        x = self.rmsnorm_layer(x)
        return x



# import numpy as np
class RotaryEmbeddings(nn.Module):
    def __init__(
        self,
         device,
        embeddings_dims: int = ModelArgs.embeddings_dims,
        block_size: int = ModelArgs.block_size,
        batch_size: int = ModelArgs.batch_size
    ):
        super().__init__()

        self.embeddings_dims = embeddings_dims
        self.block_size = block_size
        self.batch_size = batch_size
        self.theta = 0



        self.device=device

    def init_matrix(self, seq_len):
        self.matrix = torch.zeros((seq_len, self.embeddings_dims, self.embeddings_dims), dtype=torch.float32,  requires_grad=False,  device = self.device)

        positions = torch.arange(seq_len,  dtype=torch.float32,  device = self.device).unsqueeze(1)
        # dims = torch.arange(1, self.embeddings_dims // 2,  dtype=torch.float32)
        theta = 10000 ** (-2 * (positions - 1) / self.embeddings_dims)
        angles = positions * theta

        cos_angles = torch.cos(angles)
        sin_angles = torch.sin(angles)

        indices = torch.arange(self.embeddings_dims,  dtype=torch.int64,  device = self.device)
        # print(indices)
        # print(indices.shape)
        # print(indices[::2])
        even_indices = indices[::2]
        odd_indices = indices[1::2]

        self.matrix[:, even_indices, even_indices] = cos_angles
        self.matrix[:, odd_indices, odd_indices] = sin_angles
        self.matrix[:, odd_indices, even_indices] = -sin_angles
        self.matrix[:, even_indices, odd_indices] = cos_angles

        return self.matrix

    def forward(self, x):
        # B,T,C = x.shape
        # print("MATRIX:",x)
        if(x > self.block_size or x < self.block_size):
            matrix = self.init_matrix(x)
            return matrix
        else:
            matrix = self.init_matrix(self.block_size)

            return matrix


class RotaryAttentionHead(nn.Module):
    def __init__(
        self,
         device,
        embeddings_dims: int = ModelArgs.embeddings_dims,
        no_of_heads: int = ModelArgs.no_of_heads,
        attn_dropout: int = ModelArgs.attn_dropout
    ):
        super().__init__()
        self.head_size = embeddings_dims // no_of_heads
        self.query = nn.Linear(in_features=embeddings_dims, out_features=embeddings_dims,  bias=False, dtype=torch.float32,  device = device)
        self.key = nn.Linear(in_features=embeddings_dims, out_features=embeddings_dims,  bias=False, dtype=torch.float32,  device = device)
        self.value = nn.Linear(in_features=embeddings_dims, out_features=embeddings_dims,  bias=False, dtype=torch.float32,  device = device)
        self.rotary_matrix = RotaryEmbeddings(embeddings_dims=embeddings_dims,  device = device)
        self.dropout = nn.Dropout(p = attn_dropout)
        self.device = device
    def forward(self,x):
        # print(x.shape)
        batch, block_size, embeddings_dims = x.shape
        query = self.query(x)
        # print(query)
        key = self.key(x)
        values = self.value(x)
        matrix = self.rotary_matrix(block_size)

        # print(matrix.shape)
        # print(query.shape)
        masked = torch.tril(torch.ones((block_size, block_size),  requires_grad=False,  device = self.device))
        rotary_query = matrix @ query.permute(1,2,0) # (B,T, C,C) @ (B,T,C) -> (B,C,T) = (B,T,C,T)
        rotary_key = matrix @ key.permute(1,2,0)  #  (B,T, C,C  ) @ (B,T,C) -> (B,C,T) = (B,T,C,T)
        weights = rotary_query.permute(2,0,1) @ rotary_key.permute(2,0,1).transpose(-2, -1)#(B,T,C,T) @ (B,T,C,T) = (T,C,C,T)
        weights_masked = weights.masked_fill(masked == 0, float('-inf'))
        scaled_weights = weights_masked / (torch.sqrt(torch.tensor(key.shape[-1])))
        scaled_weights = F.softmax(scaled_weights, dim=-1)
        value = scaled_weights @ values
        out = self.dropout(value)
        return out


class MQA(nn.Module):
    def __init__(
        self,
        device,
        embeddings_dims: int = ModelArgs.embeddings_dims,
        block_size: int = ModelArgs.block_size,
        no_of_kv_heads: int = ModelArgs.no_of_heads,
        no_of_heads: int = ModelArgs.no_of_heads,

    ):
        super().__init__()

        self.no_of_kv_heads = no_of_kv_heads
        self.no_of_q_heads = no_of_heads // no_of_kv_heads
        self.head_size = embeddings_dims // self.no_of_q_heads
        self.rotary_matrix = RotaryEmbeddings(embeddings_dims=embeddings_dims,  device = device)
        # self.query = nn.Linear(in_features=embeddings_dims, out_features=self.head_size,  bias=False)
        self.key = nn.Linear(in_features=embeddings_dims, out_features=embeddings_dims,  dtype=torch.float32, bias=False,  device = device)
        self.value = nn.Linear(in_features=embeddings_dims, out_features=embeddings_dims,  dtype=torch.float32, bias=False,  device = device)
        self.dropout = nn.Dropout(p = ModelArgs.attn_dropout)
        self.linear_layer = nn.Linear(in_features=embeddings_dims, out_features=embeddings_dims,  dtype=torch.float32, bias=False,  device = device)
        self.device = device
        self.multi_query = nn.ModuleList([nn.Linear(in_features=embeddings_dims, out_features=embeddings_dims,  bias=False,  device = self.device) for _ in range(self.no_of_q_heads)])

    def scaled_dot_product(self, q, k, v, block_size, matrix):

            # masked = torch.tril(torch.ones((block_size, block_size),  requires_grad=False,  device = self.device))

            masked = torch.tril(torch.ones((block_size, block_size),  requires_grad=False,  device = self.device))
            rotary_query = matrix @ q.permute(1,2,0) # (B,T, C,C) @ (B,T,C) -> (B,C,T) = (B,T,C,T)
            rotary_key = matrix @ k.permute(1,2,0)  #  (B,T, C,C  ) @ (B,T,C) -> (B,C,T) = (B,T,C,T)
            weights = rotary_query.permute(2,0,1) @ rotary_key.permute(2,0,1).transpose(-2, -1)#(B,T,C,T) @ (B,T,C,T) = (T,C,C,T)
            weights_masked = weights.masked_fill(masked == 0, float('-inf'))
            scaled_weights = weights_masked / (torch.sqrt(torch.tensor(k.shape[-1])))
            scaled_weights = F.softmax(scaled_weights, dim=-1)
            value = scaled_weights @ v
            out = self.dropout(value)
            return value

    def forward(self,x):
        # print("MQA: ", x.shape)
        batch, block_size, embeddings_dims = x.shape

        # query = self.query(x)
        matrix = self.rotary_matrix(block_size)


        key = self.key(x)
        values = self.value(x)

        multi_query_concat = torch.cat([self.scaled_dot_product(query(x), key, values, block_size, matrix) for query in self.multi_query], dim=-1)


        linear_layer= self.linear_layer(multi_query_concat)
        out = self.dropout(linear_layer)
        return out


class GQA(nn.Module):
    def __init__(
        self,
         device,
        embeddings_dims: int = ModelArgs.embeddings_dims,
        block_size: int = ModelArgs.block_size,
        no_of_q_heads: int = ModelArgs.no_of_heads,
        no_of_kv_heads: int = ModelArgs.no_kv_heads
    ):
        super().__init__()

        self.no_of_kv_heads = no_of_kv_heads
        self.no_of_q_heads = no_of_q_heads
        self.dropout = nn.Dropout(p = ModelArgs.attn_dropout)
        self.linear_layer = nn.Linear(in_features=embeddings_dims * self.no_of_kv_heads, out_features=embeddings_dims , dtype=torch.float32,  bias=False,  device = device)
        self.device = device
        self.mqa = nn.ModuleList([MQA(embeddings_dims=embeddings_dims, device = self.device, block_size=block_size) for _ in range(self.no_of_kv_heads)])

    def forward(self,x):

        batch, block_size, embeddings_dims = x.shape


        grouped_query_concat = torch.cat([group(x) for group in self.mqa], dim=-1)

        linear_layer= self.linear_layer(grouped_query_concat)
        out = self.dropout(linear_layer)
        return out


class Swish(nn.Module):
    def __init__(
        self,
         device,
        block_size: int = ModelArgs.block_size,
        embeddings_dims: int = ModelArgs.embeddings_dims
    ):
        super().__init__()

        self.sig = torch.nn.Sigmoid()


    def forward(self, x):
        swish = x * self.sig(x)

        return swish



class SWiGLU(nn.Module):
    def __init__(
        self,
        device,
        block_size: int = ModelArgs.block_size,
        embeddings_dims: int = ModelArgs.embeddings_dims
    ):
        super().__init__()

        self.swish = Swish(block_size=block_size, embeddings_dims=embeddings_dims, device=device)
        self.linear_layer1 = nn.Linear(in_features=embeddings_dims, out_features=embeddings_dims,  bias=False, dtype=torch.float32,  device = device)
        self.linear_layer2 = nn.Linear(in_features=embeddings_dims, out_features=embeddings_dims,  bias=False, dtype=torch.float32,  device = device)
        self.linear_layer3 = nn.Linear(in_features=embeddings_dims, out_features=embeddings_dims,  bias=False, dtype=torch.float32,  device = device)




    def forward(self, x):
        swish_res = self.swish(self.linear_layer1(x))
        x_V = self.linear_layer2(x)
        res = torch.mul(swish_res, x_V)
        out = self.linear_layer3(res)
        return out



class FFN(nn.Module):
    def __init__(self,
                  device,
                  embeddings_dims: int = ModelArgs.embeddings_dims,
                  block_size: int = ModelArgs.block_size,
                  vocab_size: int = ModelArgs.vocab_size,
                   dropout = ModelArgs.dropout

                 ):
        super().__init__()

        self.linear_layer = nn.Linear(in_features=embeddings_dims, out_features=embeddings_dims,  dtype=torch.float32,  device = device)
        self.swiglue = SWiGLU(block_size=block_size, embeddings_dims=embeddings_dims,  device = device)
        self.dropout = nn.Dropout(p = dropout)
    def forward(self, x):

        x = self.swiglue(x)
        x = self.linear_layer(x)
        x = self.dropout(x)
        return x


class DecoderLayer(nn.Module):
    def __init__(self,
                  device,
                embeddings_dims: int = ModelArgs.embeddings_dims,
                dropout = ModelArgs.dropout,
                block_size: int = ModelArgs.block_size,
                vocab_size: int = ModelArgs.vocab_size,

                 ) :
        super().__init__()


        self.feedforward_network = FFN(embeddings_dims=embeddings_dims, block_size=block_size, vocab_size=vocab_size,  device = device)
        self.gqa = GQA(embeddings_dims=embeddings_dims, block_size=block_size, no_of_kv_heads=ModelArgs.no_kv_heads, no_of_q_heads=ModelArgs.no_of_heads,  device = device)
        # self.norm = Normalization(embeddings_dims=embeddings_dims)
        self.norm1 = Normalization(embeddings_dims=embeddings_dims)
        self.norm2 = Normalization(embeddings_dims=embeddings_dims)
        self.dropout = nn.Dropout(p = dropout)
    def forward(self, x):

        x = self.norm1(x + self.gqa(x))
        x = self.norm2(x + self.feedforward_network(x))
        return x


class Llama(nn.Module):
    def __init__(self,
                device,
                  embeddings_dims: int = ModelArgs.embeddings_dims,
                  no_of_decoder_layers: int = ModelArgs.no_of_decoder_layers,
                  block_size: int = ModelArgs.block_size,
                  vocab_size: int = ModelArgs.vocab_size,
                  dropout = ModelArgs.dropout

                 ) :
        super().__init__()

        self.embeddings = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embeddings_dims,  dtype=torch.float32,  device = device)
        self.decoder = nn.Sequential(*[DecoderLayer(embeddings_dims=embeddings_dims, block_size=block_size, vocab_size=vocab_size, dropout=dropout,  device = device) for _ in range(no_of_decoder_layers)])
        self.linear_layer = nn.Linear(in_features=embeddings_dims, out_features=vocab_size,  dtype=torch.float32,  device = device)
        self.dropout = nn.Dropout(p = dropout)
        # self.norm = Normalization(embeddings_dims)
    def forward(self, x):
        x = self.embeddings(x)
        x = self.dropout(x)
        x = self.decoder(x)
        # x = self.norm(x)
        x = self.linear_layer(x)
        # out = self.norm(x)
        return x

