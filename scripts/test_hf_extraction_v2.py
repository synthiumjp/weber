"""
Phase 0c: Corrected hidden state test.
Tests single-token magnitudes AND multi-token final-token extraction.

The v2.6 pre-registration specifies:
  - Single-token: extract at magnitude token position
  - Multi-token: extract at final token of magnitude expression

Run: python scripts\test_hf_extraction_v2.py
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
import time

print("Loading Mistral tokeniser...")
tok = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3")

# First: check which magnitudes are single-token
print("\n=== Tokenisation check for all 26 magnitudes ===")
magnitudes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 30, 40, 50,
              60, 70, 80, 90, 100, 150, 200, 300, 500, 700, 1000]

for mag in magnitudes:
    toks = tok.encode(str(mag), add_special_tokens=False)
    strs = [tok.decode([t]) for t in toks]
    status = "SINGLE" if len(toks) == 1 else f"MULTI({len(toks)})"
    print(f"  {mag:>5}: {status:>10}  tokens={strs}")

# Find single-token pairs for clean comparison
print("\n=== Loading model ===")
t0 = time.time()
model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-7B-Instruct-v0.3",
    torch_dtype=torch.float16,
    device_map="cpu",
    output_hidden_states=True,
)
print(f"Loaded in {time.time()-t0:.1f}s")
model.eval()

carrier = "The number {N} is a quantity."

# Test with magnitudes that should have distinct representations
test_pairs = [(1, 9), (1, 100), (5, 500), (10, 1000)]

for mag_a, mag_b in test_pairs:
    sent_a = carrier.format(N=mag_a)
    sent_b = carrier.format(N=mag_b)

    # Tokenise both
    inp_a = tok(sent_a, return_tensors="pt")
    inp_b = tok(sent_b, return_tensors="pt")
    ids_a = inp_a["input_ids"][0].tolist()
    ids_b = inp_b["input_ids"][0].tolist()
    strs_a = [tok.decode([t]) for t in ids_a]
    strs_b = [tok.decode([t]) for t in ids_b]

    # Find the magnitude token position(s)
    # Strategy: find where "is" appears — magnitude is everything between "number"/"" and "is"
    # For carrier "The number {N} is a quantity.", the structure is:
    #   <s> The number [space?] [mag tokens...] is a quantity .

    # Find "is" position
    is_pos_a = None
    is_pos_b = None
    for i, s in enumerate(strs_a):
        if s.strip() == "is":
            is_pos_a = i
            break
    for i, s in enumerate(strs_b):
        if s.strip() == "is":
            is_pos_b = i
            break

    # Last magnitude token is just before "is"
    mag_last_a = is_pos_a - 1 if is_pos_a else 4
    mag_last_b = is_pos_b - 1 if is_pos_b else 4

    print(f"\n{'='*60}")
    print(f"Comparing {mag_a} vs {mag_b}")
    print(f"  Sent A: {strs_a}")
    print(f"  Sent B: {strs_b}")
    print(f"  Mag A last token: pos {mag_last_a} = \"{strs_a[mag_last_a]}\"")
    print(f"  Mag B last token: pos {mag_last_b} = \"{strs_b[mag_last_b]}\"")

    with torch.no_grad():
        out_a = model(**inp_a)
        out_b = model(**inp_b)

    print(f"  Layer-by-layer (at final magnitude token):")
    for layer in [0, 4, 8, 12, 16, 20, 24, 28, 31, 32]:
        if layer < len(out_a.hidden_states):
            v_a = out_a.hidden_states[layer][0, mag_last_a, :].float().numpy()
            v_b = out_b.hidden_states[layer][0, mag_last_b, :].float().numpy()
            cos = np.dot(v_a, v_b) / (np.linalg.norm(v_a) * np.linalg.norm(v_b))
            euc = np.linalg.norm(v_a - v_b)
            print(f"    Layer {layer:2d}: cosine={cos:.4f}  euclidean={euc:.2f}")

    # Also check at the "is" token (post-magnitude, fully contextualised)
    if is_pos_a and is_pos_b:
        print(f"  At 'is' token (post-magnitude):")
        for layer in [0, 16, 24, 32]:
            if layer < len(out_a.hidden_states):
                v_a = out_a.hidden_states[layer][0, is_pos_a, :].float().numpy()
                v_b = out_b.hidden_states[layer][0, is_pos_b, :].float().numpy()
                cos = np.dot(v_a, v_b) / (np.linalg.norm(v_a) * np.linalg.norm(v_b))
                euc = np.linalg.norm(v_a - v_b)
                print(f"    Layer {layer:2d}: cosine={cos:.4f}  euclidean={euc:.2f}")

del model
print("\n\nDone.")
