"""Authorized Spark-X2.5 checkpoint loader. We do not commit weights."""
import argparse
from pathlib import Path
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

def main():
    p=argparse.ArgumentParser(); p.add_argument("--model",default="XHToken/Spark-X2.5-4B"); p.add_argument("--cache-dir",default="./checkpoints"); a=p.parse_args()
    Path(a.cache_dir).mkdir(parents=True,exist_ok=True)
    cfg=AutoConfig.from_pretrained(a.model,trust_remote_code=True,cache_dir=a.cache_dir)
    tok=AutoTokenizer.from_pretrained(a.model,trust_remote_code=True,cache_dir=a.cache_dir)
    model=AutoModelForCausalLM.from_pretrained(a.model,trust_remote_code=True,torch_dtype="auto",cache_dir=a.cache_dir,device_map="auto")
    print({"model":a.model,"model_type":getattr(cfg,"model_type",None),"layers":getattr(cfg,"num_hidden_layers",None),"vocab":getattr(cfg,"vocab_size",None),"parameters":sum(p.numel() for p in model.parameters()),"tokenizer":tok.__class__.__name__})
if __name__=="__main__": main()
