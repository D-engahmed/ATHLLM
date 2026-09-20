"""Quality-gated semantic corpus manifest builder; never claims raw data is high quality."""
import argparse,hashlib,json,re
from pathlib import Path
def score(t):
    w=re.findall(r"\S+",t)
    if len(w)<32:return 0.0
    rep=len(w)/max(1,len(set(w)))
    return max(0,min(1,.55+min(.25,len(w)/10000)+.10*(1-min(1,rep/20))))
def main():
    p=argparse.ArgumentParser(); p.add_argument("--input",required=True); p.add_argument("--output",required=True); p.add_argument("--min-quality",type=float,default=.85); a=p.parse_args()
    seen=set(); rows=[]
    for f in Path(a.input).rglob("*"):
        if f.suffix.lower() not in {".txt",".md",".jsonl"}: continue
        t=f.read_text(encoding="utf-8",errors="ignore")
        for c in re.split(r"\n\s*\n",t):
            c=c.strip(); h=hashlib.sha256(c.encode()).hexdigest()
            if c and h not in seen:
                seen.add(h); q=score(c)
                if q>=a.min_quality: rows.append({"sha256":h,"quality":q,"source":str(f),"text":c})
    Path(a.output).parent.mkdir(parents=True,exist_ok=True)
    with open(a.output,"w",encoding="utf-8") as out:
        for r in rows: out.write(json.dumps(r,ensure_ascii=False)+"\n")
    print(f"accepted={len(rows)} unique={len(seen)}")
if __name__=="__main__": main()
