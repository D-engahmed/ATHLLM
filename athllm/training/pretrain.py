import argparse,yaml
def main():
 p=argparse.ArgumentParser(); p.add_argument("--config",required=True); a=p.parse_args()
 with open(a.config,encoding="utf-8") as f:c=yaml.safe_load(f)
 print("ATHLLM target tokens:",c["data"]["target_tokens"],"pattern:",c["model"]["attention_pattern"])
if __name__=="__main__":main()
