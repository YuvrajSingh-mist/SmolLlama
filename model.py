
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

        # self.d_model = embeddings_dims
        # self.i = torch.arange(0, embeddings_dims, dtype=torch.float32)
        # # self.pos = torch.arange(0, block_size, dtype=torch.float32)
        # self.exp = ((2 * self.i)) / self.d_model
        # self.theta = 10000 ** self.exp
        # # print(self.theta.shape)
        # self.x_reshaped = torch.randn(batch_size, block_size, embeddings_dims,dtype=torch.float32, device=device)

        # self.cos = torch.cos((self.i / self.theta))
        # self.sin = torch.sin((self.i / self.theta))

        # self.even = self.sin[::2]
        # self.odd = self.cos[1::2]

        # # self.block = torch.empty((odd.size(0) + even.size(0),), dtype=self.even.dtype)
        # self.x_reshaped[..., : , ::2] = self.even
        # self.x_reshaped[..., : , 1::2] = self.odd

    
    def apply_rope(self, seq):
        batch_size, seq_len, embeds_dims = seq.shape
        # print(seq.shape)
        # print(self.embeddings_dims)
        # self.matrix = torch.zeros((seq_len, self.embeddings_dims, self.embeddings_dims), dtype=torch.float32,  requires_grad=False,  device = self.device)

        positions = torch.arange(0 , embeds_dims, 2, dtype=torch.float32,  device = self.device).unsqueeze(0)
        # dims = torch.arange(1, self.embeddings_dims // 2,  dtype=torch.float32)
        theta = 10000 ** (-2 * (positions) / embeds_dims)
        angles = positions * theta
        angles = angles.expand(seq_len, -1) # because this thing needs to be applied to every sequence in the batch but with embeds dims halved
        x_reshaped = seq.view(batch_size, seq_len, embeds_dims // 2, 2)
        
        cos_angles = torch.cos(angles)
        sin_angles = torch.sin(angles)
        # print(cos_angles.shape)
        # print(sin_angles.shape)
        # print(x_reshaped.shape)
        # indices = torch.arange(self.embeddings_dims,  dtype=torch.int64,  device = self.device)

        out = torch.stack([x_reshaped[..., 0]*cos_angles - (x_reshaped[...,1] * sin_angles), x_reshaped[...,1] * cos_angles + x_reshaped[..., 0] * sin_angles], dim=-1)
        out = out.view(batch_size, seq_len, embeds_dims)
        return out

    def forward(self, x):
        # print("X shape: ", x.shape)
        # print("X is: ", x)
        # B,T,C = x.shape
        # print("MATRIX:",x)
        # if(x > self.block_size or x < self.block_size):
        #     matrix = self.init_matrix(x)
        #     return matrix
        # else:
        #     matrix = self.init_matrix(self.block_size)

        #     return matrix
        # if(ModelArgs.inference):
        res = self.apply_rope(x)
        return res 
        # else:
            # return self.x_reshaped
    
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
        self.query = nn.Linear(in_features=embeddings_dims, out_features=self.head_size,  bias=False, dtype=torch.float32,  device = device)
        self.key = nn.Linear(in_features=embeddings_dims, out_features=self.head_size,  bias=False, dtype=torch.float32,  device = device)
        self.value = nn.Linear(in_features=embeddings_dims, out_features=self.head_size,  bias=False, dtype=torch.float32,  device = device)
        self.rope = RotaryEmbeddings(embeddings_dims=self.head_size,  device = device)
        self.dropout = nn.Dropout(p = attn_dropout)
        self.device = device
    def forward(self,x):
        # print(x.shape)
        # print("X is: ", x)
        batch, block_size, embeddings_dims = x.shape
        query = self.query(x)
        # print(query)
        key = self.key(x)
        values = self.value(x)
        # matrix = self.rotary_matrix(block_size)
        rotary_q = self.rope(query)
        rotary_k = self.rope(key)
        
        # print(matrix.shape)
        # print(query.shape)
        masked = torch.tril(torch.ones((block_size, block_size),  requires_grad=False,  device = self.device))
        # rotary_query = matrix @ query.permute(1,2,0) # (B,T, C,C) @ (B,T,C) -> (B,C,T) = (B,T,C,T)
        # rotary_key = matrix @ key.permute(1,2,0)  #  (B,T, C,C  ) @ (B,T,C) -> (B,C,T) = (B,T,C,T)
        weights = rotary_q.permute(2,0,1) @ rotary_k.permute(2,0,1).transpose(-2, -1)#(B,T,C,T) @ (B,T,C,T) = (T,C,C,T)
        weights_masked = weights.masked_fill(masked == 0, float('-inf'))
        scaled_weights = weights_masked / (torch.sqrt(torch.tensor(key.shape[-1])))
        scaled_weights = F.softmax(scaled_weights, dim=-1)
        value = scaled_weights @ values
        out = self.dropout(value)
        return out


# # import numpy as np
# class RotaryEmbeddings(nn.Module):
#     def __init__(
#         self,
#          device,
#         embeddings_dims: int = ModelArgs.embeddings_dims,
#         block_size: int = ModelArgs.block_size,
#         batch_size: int = ModelArgs.batch_size
#     ):
#         super().__init__()

#         self.embeddings_dims = embeddings_dims
#         self.block_size = block_size
#         self.batch_size = batch_size
#         self.theta = 0


#     # def init_matrix(self, seq_len):
#     #         self.matrix = torch.zeros((seq_len, self.embeddings_dims, self.embeddings_dims), dtype=torch.float32,  requires_grad=False)
#     #         for pos in range(seq_len):
#     #             for j in range(1, self.embeddings_dims // 2):
#     #                 self.theta = 10000 ** (-2*(pos-1) / self.embeddings_dims)
#     #                 self.matrix[pos, 2*j + 1, 2*j + 1] = np.cos((pos*self.theta))
#     #                 self.matrix[pos, 2*j + 1, j + 1] = -np.sin((pos* self.theta))
#     #                 self.matrix[pos, 2*j , 2*j ] = -np.cos((pos* self.theta))
#     #                 self.matrix[pos, 2*j + 1, 2*j + 1] = np.sin((pos* self.theta))
#     #         return self.matrix
#         self.device=device

#     def init_matrix(self, seq_len):
#         self.matrix = torch.zeros((seq_len, self.embeddings_dims, self.embeddings_dims), dtype=torch.float32,  requires_grad=False,  device = self.device)

#         positions = torch.arange(0 , seq_len, 2, dtype=torch.float32,  device = self.device).unsqueeze(1)
#         # dims = torch.arange(1, self.embeddings_dims // 2,  dtype=torch.float32)
#         theta = 10000 ** (-2 * (positions - 1) / self.embeddings_dims)
#         angles = positions * theta

#         cos_angles = torch.cos(angles)
#         sin_angles = torch.sin(angles)

#         indices = torch.arange(seq_len,  dtype=torch.int64,  device = self.device)
#         # print(indices)
#         # print(indices.shape)
#         # print(indices[::2])
#         even_indices = indices[::2]
#         odd_indices = indices[1::2]

#         self.matrix[:, even_indices, even_indices] = cos_angles
#         self.matrix[:, odd_indices, odd_indices] = sin_angles
#         self.matrix[:, odd_indices, even_indices] = -sin_angles
#         self.matrix[:, even_indices, odd_indices] = cos_angles

#         return self.matrix

#     def forward(self, x):
#         # B,T,C = x.shape
#         # print("MATRIX:",x)
#         if(x > self.block_size or x < self.block_size):
#             matrix = self.init_matrix(x)
#             return matrix
#         else:
#             matrix = self.init_matrix(self.block_size)

#             return matrix


# class RotaryAttentionHead(nn.Module):
#     def __init__(
#         self,
#          device,
#         embeddings_dims: int = ModelArgs.embeddings_dims,
#         no_of_heads: int = ModelArgs.no_of_heads,
#         attn_dropout: int = ModelArgs.attn_dropout
#     ):
#         super().__init__()
#         self.head_size = embeddings_dims // no_of_heads
#         self.query = nn.Linear(in_features=embeddings_dims, out_features=self.head_size,  bias=False, dtype=torch.float32,  device = device)
#         self.key = nn.Linear(in_features=embeddings_dims, out_features=self.head_size,  bias=False, dtype=torch.float32,  device = device)
#         self.value = nn.Linear(in_features=embeddings_dims, out_features=self.head_size,  bias=False, dtype=torch.float32,  device = device)
#         self.rotary_matrix = RotaryEmbeddings(embeddings_dims=self.head_size,  device = device)
#         self.dropout = nn.Dropout(p = attn_dropout)
#         self.device = device
#     def forward(self,x):
#         # print(x.shape)
#         batch, block_size, embeddings_dims = x.shape
#         query = self.query(x)
#         # print(query)
#         key = self.key(x)
#         values = self.value(x)
#         matrix = self.rotary_matrix(block_size)

#         # print(matrix.shape)
#         # print(query.shape)
#         masked = torch.tril(torch.ones((block_size, block_size),  requires_grad=False,  device = self.device))
#         rotary_query = matrix @ query.permute(1,2,0) # (B,T, C,C) @ (B,T,C) -> (B,C,T) = (B,T,C,T)
#         rotary_key = matrix @ key.permute(1,2,0)  #  (B,T, C,C  ) @ (B,T,C) -> (B,C,T) = (B,T,C,T)
#         weights = rotary_query.permute(2,0,1) @ rotary_key.permute(2,0,1).transpose(-2, -1)#(B,T,C,T) @ (B,T,C,T) = (T,C,C,T)
#         weights_masked = weights.masked_fill(masked == 0, float('-inf'))
#         scaled_weights = weights_masked / (torch.sqrt(torch.tensor(key.shape[-1])))
#         scaled_weights = F.softmax(scaled_weights, dim=-1)
#         value = scaled_weights @ values
#         out = self.dropout(value)
#         return out


class MQA(nn.Module):
    def __init__(
        self,
        device,
        no_of_q_heads: int,
        embeddings_dims: int = ModelArgs.embeddings_dims,
        block_size: int = ModelArgs.block_size,
        

    ):
        super().__init__()


        # self.no_of_q_heads = no_of_heads // no_of_kv_heads
        # self.no_of_q_heads = no_of_q_heads
        self.no_of_kv_heads = 2 # I want to have a kv for each pair of query heads 
        self.head_size = embeddings_dims // no_of_q_heads
        # self.kv_head_size = (embeddings_dims // self.no_of_kv_heads) * 2
        self.rotary= RotaryEmbeddings(embeddings_dims=self.head_size,  device = device)
        # self.rotary_k = RotaryEmbeddings(embeddings_dims=self.kv_head_size,  device = device)
        # self.query = nn.Linear(in_features=embeddings_dims, out_features=self.head_size,  bias=False)
        self.key = nn.Linear(in_features=embeddings_dims, out_features=self.head_size,  dtype=torch.float32, bias=False,  device = device)
        self.value = nn.Linear(in_features=embeddings_dims, out_features=self.head_size,  dtype=torch.float32, bias=False,  device = device)
        self.dropout = nn.Dropout(p = ModelArgs.attn_dropout)
        self.linear_layer = nn.Linear(in_features=self.head_size * self.no_of_kv_heads, out_features=embeddings_dims,  dtype=torch.float32, bias=False,  device = device)
        self.device = device
        self.multi_query = nn.ModuleList([nn.Linear(in_features=embeddings_dims, out_features=self.head_size,  bias=False,  device = self.device) for _ in range(self.no_of_kv_heads)])

    def scaled_dot_product(self, q, k, v, block_size):

            # masked = torch.tril(torch.ones((block_size, block_size),  requires_grad=False,  device = self.device))
            q = self.rotary(q)
            masked_table = torch.tril(torch.ones((block_size, block_size),  requires_grad=False,  device = self.device))
            # rotary_query = matrix @ q.permute(1,2,0) # (B,T, C,C) @ (B,T,C) -> (B,C,T) = (B,T,C,T)
            # rotary_key = matrix @ k.permute(1,2,0)  #  (B,T, C,C  ) @ (B,T,C) -> (B,C,T) = (B,T,C,T)
            # print("Query: ", q.shape)
            # print("Keys: ", k.shape)
            # print(q.permute(2,0,1).shape)
            # print(k.permute(2,0,1).transpose(-2, -1).shape)
            # weights = q.permute(2,0,1) @ k.permute(2,0,1).transpose(-2, -1)#(B,T,C,T) @ (B,T,C,T) = (T,C,C,T)
            # weights = q @ k.permute(2,1,0)
            # print(weights.shape)
            # print(masked.shape)
            weights = q @ torch.transpose(k, dim0=-2, dim1=-1) * (k.shape[-1] ** -0.5)
            masked_values = weights.masked_fill(masked_table[: block_size, : block_size] == 0, float('-inf'))
            weights_normalized = nn.functional.softmax(masked_values, dim=-1) #Normalize along the embeddings dimension for all the tokens
            weights_normalized = self.dropout(weights_normalized)
            out = weights_normalized @ v
            return out

    def forward(self,x):
        # print("MQA: ", x.shape)
        batch, block_size, embeddings_dims = x.shape

        # query = self.query(x)
        # matrix = self.rotary_matrix(block_size)


        key = self.key(x)
        values = self.value(x)
        # print("Keys: ", key.shape)
        # print("Values: ", values.shape)
        # rotary_value = self.rotary(values)
        rotary_key = self.rotary(key)
        multi_query_concat = torch.cat([self.scaled_dot_product(query(x), rotary_key, values, block_size) for query in self.multi_query], dim=-1)
        # print("Multi query: ", multi_query_concat.shape)

        linear_layer= self.linear_layer(multi_query_concat)
        # out = self.dropout(linear_layer)
        return linear_layer


class GQA(nn.Module):
    def __init__(
        self,
         device,
        embeddings_dims: int = ModelArgs.embeddings_dims,
        block_size: int = ModelArgs.block_size,
        # no_of_q_heads: int = ModelArgs.no_of_heads,
        mqa_heads: int = ModelArgs.no_kv_heads
    ):
        super().__init__()

        # self.no_of_kv_heads = no_of_kv_heads
        self.no_of_q_heads = ModelArgs.no_of_heads // mqa_heads
        # self.head_dim = embeddings_dims // self.no_kv_heads
        self.dropout = nn.Dropout(p = ModelArgs.attn_dropout)
        self.linear_layer = nn.Linear(in_features=embeddings_dims * self.no_of_q_heads, out_features=embeddings_dims , dtype=torch.float32,  bias=False,  device = device)
        self.device = device
        self.mqa = nn.ModuleList([MQA(no_of_q_heads=self.no_of_q_heads, embeddings_dims=embeddings_dims, device = self.device, block_size=block_size) for _ in range(self.no_of_q_heads)])
        # self.mqa = MQA(no_of_q_heads=self.no_of_q_heads, device=self.device, embeddings_dims=embeddings_dims, block_size=block_size)
    def forward(self,x):

        batch, block_size, embeddings_dims = x.shape

        # res = self.mqa(x)
        grouped_query_concat = torch.cat([group(x) for group in self.mqa], dim=-1)

        linear_layer= self.linear_layer(grouped_query_concat) #Basically MQA is made into GQA with no_of_q_heads and this class right here is just to consolidate everything into one
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
        self.hidden_dims = int(2 * ( 4 * embeddings_dims) / 3)
        self.swish = Swish(block_size=block_size, embeddings_dims=embeddings_dims, device=device)
        self.linear_layer1 = nn.Linear(in_features=embeddings_dims, out_features=self.hidden_dims,  bias=False, dtype=torch.float32,  device = device)
        self.linear_layer2 = nn.Linear(in_features=embeddings_dims, out_features=self.hidden_dims,  bias=False, dtype=torch.float32,  device = device)
        self.linear_layer3 = nn.Linear(in_features=self.hidden_dims, out_features=embeddings_dims,  bias=False, dtype=torch.float32,  device = device)




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

        # self.linear_layer = nn.Linear(in_features=embeddings_dims, out_features=embeddings_dims,  dtype=torch.float32,  device = device)
        self.swiglue = SWiGLU(block_size=block_size, embeddings_dims=embeddings_dims,  device = device)
        self.dropout = nn.Dropout(p = dropout)
    def forward(self, x):

        x = self.swiglue(x)
        # x = self.linear_layer(x)
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
        self.gqa = GQA(embeddings_dims=embeddings_dims, block_size=block_size, mqa_heads=2,  device = device)
        # self.norm = Normalization(embeddings_dims=embeddings_dims)
        self.norm1 = Normalization(embeddings_dims=embeddings_dims)
        self.norm2 = Normalization(embeddings_dims=embeddings_dims)
        self.dropout = nn.Dropout(p = dropout)
    def forward(self, x):

        x = x + self.gqa(self.norm1(x))
        x = x + self.feedforward_network(self.norm2(x))
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
        
        
        #weight tying
        self.embeddings.weight = self.linear_layer.weight
    
        self.apply(self._init_weights)

    def _init_weights(self, module):
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
               
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
               
                     
                    
    def forward(self, x):
        x = self.embeddings(x)
        x = self.dropout(x)
        x = self.decoder(x)
        # x = self.norm(x)
        x = self.linear_layer(x)
        # out = self.norm(x)
        return x