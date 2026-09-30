import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class Config:
    hidden_size: int = 512
    num_heads: int = 8
    layers: int = 2
    h_cycles: int = 3
    l_cycles: int = 6
    beta_init: float = 0.55

    def __post_init__(self):
        if min(self.hidden_size, self.num_heads, self.layers, self.h_cycles, self.l_cycles) < 1:
            raise ValueError('Model dimensions and loop counts must be positive')
        if self.hidden_size % self.num_heads or (self.hidden_size // self.num_heads) % 2:
            raise ValueError('hidden_size / num_heads must be an even integer')
        if not 0 < self.beta_init < 1:
            raise ValueError('beta_init must be between zero and one')


def initialize(tensor, std):
    with torch.no_grad():
        bound = math.erf(2 / math.sqrt(2))
        scale = std / math.sqrt(1 - 4 * math.exp(-2) / (math.sqrt(2 * math.pi) * bound))
        tensor.uniform_(-bound, bound).erfinv_().mul_(math.sqrt(2) * scale)
        return tensor.clamp_(-2 * scale, 2 * scale)


def rms_norm(x, eps=1e-5):
    value = x.float()
    return (value * torch.rsqrt(value.square().mean(-1, keepdim=True) + eps)).to(x.dtype)


class Linear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(initialize(torch.empty(out_features, in_features), in_features ** -0.5))

    def forward(self, x):
        return F.linear(x, self.weight.to(x.dtype))


class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.heads = config.num_heads
        self.head_dim = config.hidden_size // self.heads
        self.qkv = Linear(config.hidden_size, 3 * config.hidden_size)
        self.output = Linear(config.hidden_size, config.hidden_size)
        freq = 1 / (10000 ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim))
        angles = torch.outer(torch.arange(97).float(), freq).repeat(1, 2)
        self.register_buffer('cos', angles.cos()[None, :, None, :], persistent=False)
        self.register_buffer('sin', angles.sin()[None, :, None, :], persistent=False)

    def rotate(self, x):
        value = x.float()
        left, right = value.chunk(2, dim=-1)
        return (value * self.cos + torch.cat((-right, left), dim=-1) * self.sin).to(x.dtype)

    def forward(self, x):
        batch, length, width = x.shape
        q, k, v = self.qkv(x).reshape(batch, length, 3 * self.heads, self.head_dim).chunk(3, dim=2)
        q, k, v = self.rotate(q).transpose(1, 2), self.rotate(k).transpose(1, 2), v.transpose(1, 2)
        value = F.scaled_dot_product_attention(q, k, v, scale=self.head_dim ** -0.5)
        return self.output(value.transpose(1, 2).reshape(batch, length, width))


class Layer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = Attention(config)
        intermediate = math.ceil(round(4 * config.hidden_size * 2 / 3) / 256) * 256
        self.gate_up = Linear(config.hidden_size, 2 * intermediate)
        self.down = Linear(intermediate, config.hidden_size)

    def forward(self, x):
        x = rms_norm(x + self.attention(x))
        gate, value = self.gate_up(x).chunk(2, dim=-1)
        return rms_norm(x + self.down(F.silu(gate) * value))


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layers = nn.ModuleList([Layer(config) for _ in range(config.layers)])
        self.conv = nn.Conv2d(config.hidden_size, config.hidden_size, 3, padding=1,
                              groups=config.hidden_size, bias=False)

    def forward(self, state, injection):
        batch, _, width = state.shape
        grid = state[:, 16:].reshape(batch, 9, 9, width).permute(0, 3, 1, 2).contiguous()
        grid = F.conv2d(grid, self.conv.weight.to(state.dtype), padding=1, groups=width)
        grid = grid.permute(0, 2, 3, 1).reshape(batch, 81, width).contiguous()
        x = torch.cat((state[:, :16], grid), dim=1) + injection
        for layer in self.layers:
            x = layer(x)
        return x


class Connection(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1, config.hidden_size))
        self.beta_logit = nn.Parameter(torch.logit(torch.tensor([config.beta_init])))

    def score(self, x):
        return torch.einsum('hd,btd->bth', self.weight.to(x.dtype), rms_norm(x, 1e-6)).float().clamp(-8, 8).exp()

    def start(self, x):
        weight = self.score(x)
        return weight * x.float(), weight

    def forward(self, x, memory):
        numerator, denominator = memory
        weight = self.score(x)
        beta = self.beta_logit.sigmoid().float()
        numerator = beta * numerator + weight * x.float()
        denominator = beta * denominator + weight
        return (numerator / denominator).to(x.dtype), (numerator, denominator)


class InfiLoop(nn.Module):
    def __init__(self, config, dtype=torch.float32):
        super().__init__()
        self.config = config
        self.dtype = dtype
        self.embedding = nn.Parameter(initialize(torch.empty(11, config.hidden_size), config.hidden_size ** -0.5))
        self.head = Linear(config.hidden_size, 11)
        self.prefix = nn.Parameter(torch.zeros(1, config.hidden_size))
        self.block = Block(config)
        self.inner = Connection(config)
        self.outer = Connection(config)
        self.register_buffer('initial_h', initialize(torch.empty(config.hidden_size, dtype=dtype), 1))
        self.register_buffer('initial_l', initialize(torch.empty(config.hidden_size, dtype=dtype), 1))

    def initial_state(self, batch_size):
        shape = (batch_size, 97, self.config.hidden_size)
        high, low = self.initial_h.expand(shape), self.initial_l.expand(shape)
        with torch.no_grad():
            numerator, denominator = self.outer.start(high)
        return high, low, numerator, denominator

    def cycle(self, high, low, memory, embedding):
        inner_memory = self.inner.start(low)
        for _ in range(self.config.l_cycles):
            candidate = self.block(low, high + embedding)
            low, inner_memory = self.inner(candidate, inner_memory)
        high, memory = self.outer(self.block(high, low), memory)
        return high, low, memory

    def forward(self, inputs, state=None):
        if state is None:
            state = self.initial_state(len(inputs))
        high, low, numerator, denominator = state
        embedding = F.embedding(inputs, self.embedding.to(self.dtype))
        prefix = F.pad(self.prefix.expand(len(inputs), -1).to(self.dtype), (0, 15 * self.config.hidden_size))
        prefix = prefix.reshape(len(inputs), 16, self.config.hidden_size)
        embedding = math.sqrt(self.config.hidden_size) * torch.cat((prefix, embedding), dim=1)
        memory = (numerator, denominator)
        with torch.no_grad():
            for _ in range(self.config.h_cycles - 1):
                high, low, memory = self.cycle(high, low, memory, embedding)
        high, low, memory = self.cycle(high, low, memory, embedding)
        logits = self.head(high)[:, 16:]
        return logits, tuple(x.detach() for x in (high, low, *memory))


def loss_function(logits, labels):
    x = logits.double()
    positive = torch.where(x < 0, 1 / (1 - x + 1e-30), x + 1)
    log_probability = (positive / positive.sum(-1, keepdim=True)).log()
    return -log_probability.gather(-1, labels.unsqueeze(-1)).mean()
