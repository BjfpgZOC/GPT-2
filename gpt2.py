from dataclasses import dataclass

import math
import inspect

import torch
import torch.nn as nn
from torch.nn import functional as F

from transformers import GPT2LMHeadModel

# ------------------------------------ CONFIG ------------------------------------

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embed: int = 768

# ------------------------------------ MODEL ------------------------------------

class MLP(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embed, 4 * config.n_embed)
        self.gelu = nn.GELU(approximate = 'tanh')
        self.c_proj = nn.Linear(4 * config.n_embed, config.n_embed)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        
    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

class CausalAttention(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embed % config.n_head == 0

        self.c_attn = nn.Linear(config.n_embed, 3 * config.n_embed)
        self.c_proj = nn.Linear(config.n_embed, config.n_embed)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        self.n_head = config.n_head
        self.n_embed = config.n_embed

        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size))

    def forward(self, x, fla = True):
        B, T, C = x.size()

        q, k, v = self.c_attn(x).split(self.n_embed, dim = 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)             # (B, T, n_head, dh) -> (B, n_head, T, dh)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)             # (B, T, n_head, dh) -> (B, n_head, T, dh)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)             # (B, T, n_head, dh) -> (B, n_head, T, dh)
        
        if fla:
            y = F.scaled_dot_product_attention(q, k, v, is_causal = True)
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))             # (B, n_head, T, dh) @ (B, n_head, dh, T) -> (B, n_head, T, T)
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
            att = F.softmax(att, dim = -1)
            y = att @ v                                                                 # (B, n_head, T, T) @ (B, n_head, T, dh) -> (B, n_head, T, dh)

        y = y.transpose(1, 2).contiguous().view(B, T, C)                            # (B, n_head, T, dh) -> (B, T, n_head, dh) -> (B, T, n_head * dh)
        y = self.c_proj(y)
        return y

class Block(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embed)
        self.attn = CausalAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embed)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embed),
            wpe = nn.Embedding(config.block_size, config.n_embed),
            h = nn.ModuleList(Block(config) for _ in range(config.n_layer)),
            ln_f = nn.LayerNorm(config.n_embed),
        ))

        self.lm_head = nn.Linear(config.n_embed, config.vocab_size, bias = False)

        self.transformer.wte.weight = self.lm_head.weight

        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean = 0.0, std = std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean = 0.0, std = 0.02)

    def configure_optimizers(self, weight_decay, learning_rate, device_type):
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}

        decay_params = [p for n, p in param_dict.items() if p.dim() >=2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0}
        ]

        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        
        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        print(f"using fused AdamW: {use_fused}")

        optimizer =  torch.optim.AdamW(optim_groups, lr = learning_rate, betas = (0.9, 0.95), eps = 1e-8, fused = use_fused)
        return optimizer
    
    def forward(self, idx, targets = None):
        B, T = idx.size()

        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is {self.config.block_size}."

        pos = torch.arange(0, T, dtype = torch.long, device = idx.device)           # (T)
        pos_emb = self.transformer.wpe(pos)                                         # (T, n_embed)
        tok_emb = self.transformer.wte(idx)                                         # (B, T, n_embed)
        x = pos_emb + tok_emb                                                       # (B, T, n_embed)

        for block in self.transformer.h:
            x = block(x)                                                            # (B, T, n_embed)

        x = self.transformer.ln_f(x)                                                # (B, T, n_embed)
        logits = self.lm_head(x)                                                    # (B, T, vocab_size)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss
    
    @classmethod
    def from_pretrained(cls, model_type):
        assert model_type in {"gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"}
        
        print(f"Loading weights from pretrained GPT2 Model: {model_type}")

        config_args = {
            "gpt2":             dict(n_layer = 12, n_head = 12, n_embed = 768),
            "gpt2-medium":      dict(n_layer = 24, n_head = 16, n_embed = 1024),
            "gpt2-large":       dict(n_layer = 36, n_head = 20, n_embed = 1280),
            "gpt2-xl":          dict(n_layer = 48, n_head = 25, n_embed = 1600),
        }[model_type]

        config_args["vocab_size"] = 50257
        config_args["block_size"] = 1024

        config = GPTConfig(**config_args)
        model = GPT(config)

        model_state = model.state_dict()
        model_state_keys = model_state.keys()
        model_state_keys = [key for key in model_state_keys if not key.endswith(".attn.bias")]

        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        model_state_hf = model_hf.state_dict()
        model_state_hf_keys = model_state_hf.keys()
        model_state_hf_keys = [key for key in model_state_hf_keys if not key.endswith(".attn.masked_bias")]
        model_state_hf_keys = [key for key in model_state_hf_keys if not key.endswith(".attn.bias")]
        transposed = ["attn.c_attn.weight", "attn.c_proj.weight", "mlp.c_fc.weight", "mlp.c_proj.weight"]

        assert len(model_state_keys) == len(model_state_hf_keys), f"Mismatched model state keys: {len(model_state_keys)} != {len(model_state_hf_keys)}"

        for key in model_state_hf_keys:
            if any(key.endswith(w) for w in transposed):
                assert model_state_hf[key].shape[::-1] == model_state[key].shape
                with torch.no_grad():
                    model_state[key].copy_(model_state_hf[key].t())
            else:
                assert model_state_hf[key].shape == model_state[key].shape
                with torch.no_grad():
                    model_state[key].copy_(model_state_hf[key])

        return model

