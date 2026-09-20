"""Spark-X2.5-inspired ATHLLM 4B foundation.

The upstream release documents a 3 sliding-window : 1 full-attention pattern.
This implementation is a research foundation; exact checkpoint compatibility
must be validated tensor-by-tensor by the conversion layer.
"""
from dataclasses import dataclass
import math, torch
from torch import nn

@dataclass
class Spark25Config:
    vocab_size:int=131072; hidden_size:int=2560; intermediate_size:int=10240
    num_hidden_layers:int=36; num_attention_heads:int=16; num_key_value_heads:int=4
    max_position_embeddings:int=1048576; sliding_window:int=512
    rope_theta_sliding:float=10000.0; rope_theta_full:float=5000000.0
    partial_rotary_factor_full:float=0.25; rms_norm_eps:float=1e-6
    tie_word_embeddings:bool=True

class RMSNorm(nn.Module):
    def __init__(self,d,eps=1e-6): super().__init__(); self.weight=nn.Parameter(torch.ones(d)); self.eps=eps
    def forward(self,x): return x*torch.rsqrt(x.pow(2).mean(-1,keepdim=True)+self.eps)*self.weight

class GatedMLP(nn.Module):
    def __init__(self,c):
        super().__init__(); self.gate=nn.Linear(c.hidden_size,c.intermediate_size,bias=False); self.up=nn.Linear(c.hidden_size,c.intermediate_size,bias=False); self.down=nn.Linear(c.intermediate_size,c.hidden_size,bias=False)
    def forward(self,x): return self.down(torch.nn.functional.gelu(self.gate(x),approximate="none")*self.up(x))

class GQAAttention(nn.Module):
    def __init__(self,c,full=False):
        super().__init__(); self.h=c.num_attention_heads; self.kv=c.num_key_value_heads; self.d=c.hidden_size//c.num_attention_heads; self.full=full; self.window=c.sliding_window
        self.qkv=nn.Linear(c.hidden_size,(self.h+2*self.kv)*self.d,bias=False); self.o=nn.Linear(c.hidden_size,c.hidden_size,bias=False)
        self.head_gate=nn.Linear(c.hidden_size,self.h,bias=True)
    def forward(self,x):
        b,n,_=x.shape; z=self.qkv(x); q=z[...,:self.h*self.d].view(b,n,self.h,self.d).transpose(1,2)
        k=z[...,self.h*self.d:(self.h+self.kv)*self.d].view(b,n,self.kv,self.d).transpose(1,2)
        v=z[...,(self.h+self.kv)*self.d:].view(b,n,self.kv,self.d).transpose(1,2)
        r=self.h//self.kv; k=k.repeat_interleave(r,1); v=v.repeat_interleave(r,1)
        s=q@k.transpose(-2,-1)/math.sqrt(self.d)
        mask=torch.ones((n,n),dtype=torch.bool,device=x.device).tril()
        if not self.full:
            idx=torch.arange(n,device=x.device); mask &= idx[None,:]>=idx[:,None]-self.window+1
        s=s.masked_fill(~mask[None,None],torch.finfo(s.dtype).min)
        a=s.softmax(-1); y=(a@v).transpose(1,2).reshape(b,n,-1)
        y=y.view(b,n,self.h,self.d)*torch.sigmoid(self.head_gate(x)).unsqueeze(-1)
        return self.o(y.reshape(b,n,-1))

class Block(nn.Module):
    def __init__(self,c,full): super().__init__(); self.n1=RMSNorm(c.hidden_size,c.rms_norm_eps); self.a=GQAAttention(c,full); self.n2=RMSNorm(c.hidden_size,c.rms_norm_eps); self.m=GatedMLP(c)
    def forward(self,x): return x+self.a(self.n1(x)) if False else x+self.a(self.n1(x))+self.m(self.n2(x+self.a(self.n1(x))))

class ATHLLMSpark25(nn.Module):
    def __init__(self,c=None):
        super().__init__(); self.cfg=c or Spark25Config(); c=self.cfg
        self.embed_tokens=nn.Embedding(c.vocab_size,c.hidden_size)
        self.layers=nn.ModuleList([Block(c,i%4==3) for i in range(c.num_hidden_layers)])
        self.norm=RMSNorm(c.hidden_size,c.rms_norm_eps); self.lm_head=nn.Linear(c.hidden_size,c.vocab_size,bias=False)
        if c.tie_word_embeddings: self.lm_head.weight=self.embed_tokens.weight
    def forward(self,input_ids):
        x=self.embed_tokens(input_ids)
        for layer in self.layers: x=layer(x)
        return self.lm_head(self.norm(x))

if __name__=="__main__":
    c=Spark25Config(vocab_size=4096,hidden_size=256,intermediate_size=1024,num_hidden_layers=4)
    m=ATHLLMSpark25(c); y=m(torch.randint(0,c.vocab_size,(1,32))); print(y.shape,sum(p.numel() for p in m.parameters()))
