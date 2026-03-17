import torch, numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

tok = AutoTokenizer.from_pretrained('meta-llama/Meta-Llama-3-8B-Instruct')
model = AutoModelForCausalLM.from_pretrained('meta-llama/Meta-Llama-3-8B-Instruct', torch_dtype=torch.float16).to('cuda')
model.config.output_hidden_states = True
model.eval()

inputs = tok("The number 50 is a quantity.", return_tensors="pt")
inputs = {k: v.to('cuda') for k, v in inputs.items()}
with torch.no_grad():
    out = model(**inputs)

print(f'hidden_states length: {len(out.hidden_states)}')
print(f'Shape of [0]: {out.hidden_states[0].shape}')
print(f'Shape of [1]: {out.hidden_states[1].shape}')
print(f'Shape of [-1]: {out.hidden_states[-1].shape}')

# Check if layers differ
h0 = out.hidden_states[1][0, 4, :].cpu().float().numpy()
h16 = out.hidden_states[17][0, 4, :].cpu().float().numpy()
h31 = out.hidden_states[32][0, 4, :].cpu().float().numpy()
print(f'L0 vs L16 cosine: {np.dot(h0,h16)/(np.linalg.norm(h0)*np.linalg.norm(h16)):.4f}')
print(f'L0 vs L31 cosine: {np.dot(h0,h31)/(np.linalg.norm(h0)*np.linalg.norm(h31)):.4f}')
print(f'L0 norm: {np.linalg.norm(h0):.2f}, L16 norm: {np.linalg.norm(h16):.2f}, L31 norm: {np.linalg.norm(h31):.2f}')
