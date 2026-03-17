import torch, numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

tok = AutoTokenizer.from_pretrained('meta-llama/Meta-Llama-3-8B-Instruct')
model = AutoModelForCausalLM.from_pretrained('meta-llama/Meta-Llama-3-8B-Instruct', torch_dtype=torch.float16).to('cuda')
model.config.output_hidden_states = True
model.eval()

# Extract 2 sentences to test
sents = ["The number 70 is a quantity.", "The number 30 is a quantity."]
mags = ["70", "30"]

all_hs = []
for sent, mag in zip(sents, mags):
    enc = tok(sent, return_tensors="pt", return_offsets_mapping=True, add_special_tokens=False)
    offsets = enc["offset_mapping"][0].tolist()
    cs = sent.find(mag)
    ce = cs + len(mag)
    toks = [i for i, (s,e) in enumerate(offsets) if s < ce and e > cs]
    pos = toks[-1]
    
    inputs = tok(sent, return_tensors="pt")
    inputs = {k: v.to('cuda') for k, v in inputs.items()}
    with torch.no_grad():
        out = model(**inputs)
    
    n_layers = len(out.hidden_states) - 1
    hs = torch.stack([out.hidden_states[l+1][0, pos, :] for l in range(n_layers)])
    all_hs.append(hs.cpu().numpy())
    print(f'{mag}: pos={pos}, hs shape={hs.shape}, L0 norm={np.linalg.norm(hs[0].cpu().numpy()):.2f}, L16 norm={np.linalg.norm(hs[16].cpu().numpy()):.2f}')

arr = np.array(all_hs)
print(f'\nStacked shape: {arr.shape}')
print(f'L0  cosine dist: {1 - np.dot(arr[0,0], arr[1,0])/(np.linalg.norm(arr[0,0])*np.linalg.norm(arr[1,0])):.4f}')
print(f'L16 cosine dist: {1 - np.dot(arr[0,16], arr[1,16])/(np.linalg.norm(arr[0,16])*np.linalg.norm(arr[1,16])):.4f}')
print(f'L31 cosine dist: {1 - np.dot(arr[0,31], arr[1,31])/(np.linalg.norm(arr[0,31])*np.linalg.norm(arr[1,31])):.4f}')
