"""
Phase 0b: Confirm HuggingFace per-layer hidden state extraction.
Run: python scripts\test_hf_extraction.py
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
import time

print("Loading Mistral tokeniser...")
tok = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3")

print("Loading model (float16, CPU)...")
t0 = time.time()
model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-7B-Instruct-v0.3",
    torch_dtype=torch.float16,
    device_map="cpu",
    output_hidden_states=True,
)
print(f"Loaded in {time.time()-t0:.1f}s")
model.eval()

s1 = "The number 1 is a quantity."
s2 = "The number 1000 is a quantity."

for sent in [s1, s2]:
    inputs = tok(sent, return_tensors="pt")
    ids = inputs["input_ids"][0].tolist()
    strs = [tok.decode([t]) for t in ids]
    print(f'\n"{sent}"')
    print(f"  Tokens ({len(ids)}): {strs}")

print("\nForward pass (sentence 1)...")
t0 = time.time()
with torch.no_grad():
    out1 = model(**tok(s1, return_tensors="pt"))
print(f"  Done in {time.time()-t0:.1f}s")
print(f"  Layers: {len(out1.hidden_states)} (including embedding)")

print("\nForward pass (sentence 2)...")
with torch.no_grad():
    out2 = model(**tok(s2, return_tensors="pt"))

# Find the magnitude token position
ids1 = tok(s1, return_tensors="pt")["input_ids"][0].tolist()
strs1 = [tok.decode([t]) for t in ids1]
print(f"\nToken strings: {strs1}")

for pos in range(len(ids1)):
    print(f'  pos {pos}: "{strs1[pos]}"')

# Find the "1" token
mag_pos = None
for i, s in enumerate(strs1):
    if "1" in s and i > 1:
        mag_pos = i
        break
print(f'\nMagnitude token position: {mag_pos} ("{strs1[mag_pos]}")')

print(f'\nLayer-by-layer cosine similarity (pos {mag_pos}, "1" vs "1000"):')
for layer in [0, 8, 16, 20, 24, 28, 31, 32]:
    if layer < len(out1.hidden_states):
        v1 = out1.hidden_states[layer][0, mag_pos, :].float().numpy()
        v2 = out2.hidden_states[layer][0, mag_pos, :].float().numpy()
        cos = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        euc = np.linalg.norm(v1 - v2)
        print(f"  Layer {layer:2d}: cosine={cos:.4f}  euclidean={euc:.2f}")

del model
print("\nDone. HuggingFace extraction confirmed working.")
