import json
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained('meta-llama/Meta-Llama-3-8B-Instruct')
mags = [1,2,3,4,5,6,7,8,9,10,15,20,30,40,50,60,70,80,90,100,150,200,300,500,700,1000]
for m in mags:
    tokens = tok.encode(str(m), add_special_tokens=False)
    n = len(tokens)
    if n > 1:
        print(f'  {m}: {n} tokens -> {[tok.decode([t]) for t in tokens]}')
print(f'\nSingle-token: {sum(1 for m in mags if len(tok.encode(str(m), add_special_tokens=False))==1)}/26')
