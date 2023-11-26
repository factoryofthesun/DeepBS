### Class constructor for DeepBS model (cross attention + MLP)
import torch
import torch.nn as nn
from torch import einsum
from einops import rearrange

class DeepBS(nn.Module):

    def __init__(self, input_dim, qk_dim = 256, v_dim = 256, hidden_dim=128, ca_blocks=3, mlp_blocks=3, pooling='pre',
                 last_activation='sigmoid'):
        """
        Construct a DiffusionNet.

        Parameters:
            input_dim (int): dimension of input tokens
            qk_dim (int): dimension of query and key tokens
            v_dim (int): dimension of value tokens
            ca_blocks (int): number of cross attention blocks
            mlp_blocks (int): number of MLP blocks
            pooling (str): pooling method, 'pre' or 'post' the MLP pass
            last_activation (str): activation function for the last layer (cast to probs)
        """
        super(DeepBS, self).__init__()

        ## Store parameters
        # Basic parameters
        self.input_dim = input_dim
        self.qk_dim = qk_dim
        self.v_dim = v_dim
        self.ca_blocks = ca_blocks
        self.mlp_blocks = mlp_blocks
        self.pooling = pooling
        self.last_activation = last_activation

        ## Set up the network

        # Cross attention blocks
        self.ca_blocks = []
        for i_block in range(self.ca_blocks):
            block = CrossAttentionBlock(input_dim = input_dim,
                                        qk_dim = qk_dim,
                                        v_dim = v_dim)

            self.ca_blocks.append(block)
        self.ca_blocks = nn.ModuleList(self.ca_blocks)

        # MLP blocks
        self.mlp_blocks = [nn.Linear(v_dim, hidden_dim)]
        for i_block in range(self.mlp_blocks):
            block = nn.Linear(hidden_dim, hidden_dim)
            self.mlp_blocks.append(block)
        self.mlp_blocks.append(nn.Linear(hidden_dim, 1))
        self.mlp_blocks = nn.ModuleList(self.mlp_blocks)

        # Final activation
        if self.last_activation == 'sigmoid':
            self.final_layer = nn.Sigmoid()

    def forward(self, x1, x2):
        """
        Forward pass of the model.

        Parameters:
            x1 (tensor): Team 1 input features, dimension [N1,C] or [B,N1,C]
            x2 (tensor): Team 2 input features, dimension [N2,C] or [B,N2,C]
        Returns:
            x_out (tensor):    Output with dimension [B]
        """
        assert len(x1.shape) == len(x2.shape), f"Input tensors must have same number of dimensions. {x1.shape} vs {x2.shape}"
        if len(x1.shape) == 2:
            x1 = x1.unsqueeze(0)
            x2 = x2.unsqueeze(0)

        for ca_block in self.ca_blocks:
            x1, x2 = ca_block(x1, x2)

        # TODO: how to do n-layer cross attention??
        if self.pooling == 'pre':
            x = x1.mean(dim = 1)
        else:
            x = x1

        for mlp_block in self.mlp_blocks:
            x = mlp_block(x)

        if self.pooling == 'post':
            x = x.mean(dim = 1)

        # Final activation
        x_out = self.final_layer(x1).squeeze() # B

        return x_out

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = dots.softmax(dim=-1)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out =  self.to_out(out)
        return out

class CrossAttention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_k = nn.Linear(dim, inner_dim , bias=False)
        self.to_v = nn.Linear(dim, inner_dim , bias = False)
        self.to_q = nn.Linear(dim, inner_dim, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x_qkv):
        b, n, _, h = *x_qkv.shape, self.heads

        k = self.to_k(x_qkv)
        k = rearrange(k, 'b n (h d) -> b h n d', h = h)

        v = self.to_v(x_qkv)
        v = rearrange(v, 'b n (h d) -> b h n d', h = h)

        q = self.to_q(x_qkv[:, 0].unsqueeze(1))
        q = rearrange(q, 'b n (h d) -> b h n d', h = h)



        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = dots.softmax(dim=-1)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out =  self.to_out(out)
        return out


