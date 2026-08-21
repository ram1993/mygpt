import torch
import torch.nn as nn
import math


class LayerNorm(nn.Module):

    def __init__(self, emb_dim):
        super().__init__()

        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self,x):

        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)

        x_norm = (x-mean)/torch.sqrt(var + self.eps)

        return self.scale*x_norm + self.shift

class GELU(nn.Module):

    def __init__(self):
        super().__init__()

        self.coeff = math.sqrt(2/math.pi)

    def forward(self,x):

        return 0.5 * x * (1.0 + torch.tanh(self.coeff * (x+0.044715*x.pow(3))))


class FeedForward(nn.Module):

    def __init__(self, emb_dim):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(emb_dim, emb_dim*4),
            GELU(),
            nn.Linear(emb_dim*4, emb_dim)
        )

    def forward(self,x):

        return self.layers(x)


class CausalMultiHeadAttention(nn.Module):

    def __init__(self, d_in, d_out, n_heads, context_len, drop_rate=0.1, qkv_bias=False):
        super().__init__()

        assert d_out%n_heads==0, "d_out must be divisable by n_heads"

        self.head_dim = d_out//n_heads
        self.n_heads = n_heads

        self.Q_proj = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.K_proj = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.V_proj = nn.Linear(d_in, d_out, bias=qkv_bias)

        self.dropout = nn.Dropout(drop_rate)

        self.head_proj = nn.Linear(d_out, d_in, bias=qkv_bias)

        self.scale = math.sqrt(d_out//n_heads)

        self.register_buffer(
            "mask",
            torch.tril(torch.ones(context_len,context_len)).bool()

        )

    def forward(self,x):

        batch_size,seq_len, emb_dim = x.shape

        Q = self.Q_proj(x).reshape(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1,2)
        K = self.K_proj(x).reshape(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1,2)
        V = self.V_proj(x).reshape(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1,2)

        attention_weight = Q @ K.transpose(-2,-1) / self.scale

        mask = self.mask[:seq_len,:seq_len]

        attention_weight = attention_weight.masked_fill(~mask,float("-inf"))


        attention_scores = torch.softmax(attention_weight, dim=-1)

        attention_scores = self.dropout(attention_scores)

        attention_out = attention_scores @ V
        
        attention_out = attention_out.transpose(1,2).reshape(batch_size, seq_len,-1)

        out = self.head_proj(attention_out)

        return out

class TransformerBlock(nn.Module):

    def __init__(self, config):
        super().__init__()


        self.layer_norm1 = LayerNorm(config["embedding_dim"])
        self.mha = CausalMultiHeadAttention(config["embedding_dim"], config["embedding_dim"], config["n_head"], config["context_len"], config["drop_rate"], config["qkv_bias"])
        self.layer_norm2 = LayerNorm(config["embedding_dim"])
        self.feedforward = FeedForward(config["embedding_dim"])
        self.dropout = nn.Dropout(config["drop_rate"])


    def forward(self,x):
        shortcut = x 
        x = self.layer_norm1(x)
        x = self.mha(x)
        x = self.dropout(x)
        x = shortcut + x

        shortcut = x

        x = self.layer_norm2(x)
        x = self.feedforward(x)
        x = self.dropout(x)
        x = x+shortcut

        return x

class Lumiere(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.token_embedding = nn.Embedding(config["vocab_size"], config["embedding_dim"])
        self.pos_embedding = nn.Embedding(config["context_len"], config["embedding_dim"])

        self.blocks = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config["n_layers"])]
        )

        self.final_layer_norm = LayerNorm(config["embedding_dim"])

        self.out_head = nn.Linear(config["embedding_dim"], config["vocab_size"])

        self.dropout = nn.Dropout(config["drop_rate"])

    def forward(self,x):
        batch_size, seq_len = x.shape

        x_enc = self.token_embedding(x)
        x_pos = self.pos_embedding(torch.arange(seq_len, device=x.device))
        x = x_enc + x_pos 

        x = self.dropout(x)

        for block in self.blocks:

            x = block(x)
            
        x = self.final_layer_norm(x)
        logits = self.out_head(x)
        return logits


def generate_text(input_encoded, model, max_output_len=3, max_context_len=10):
    device = next(model.parameters()).device
    input_encoded = input_encoded.to(device) 
    
    model.eval()

    with torch.inference_mode():

        for _ in range(max_output_len):
            logits = model(input_encoded[:, -max_context_len:])
            # print(logits.shape, logits[:,-1].shape)
            pred_id = torch.argmax(logits[:,-1,:], dim=1, keepdim=True)
            # print(pred_id)
            input_encoded = torch.cat((input_encoded,pred_id), dim=1)

    return input_encoded
