"""
Phase 0d: Test whether llama_get_embeddings_ith returns contextualised
representations (different for different inputs) or static embeddings
(identical lookup table values).

This is the critical test. If contextualised: we can use llama-cpp for
EVERYTHING on Vulkan GPU. No HuggingFace needed. Entire project runs
in <20 min per model.

Run: python scripts\test_llama_contextualised.py
"""
import ctypes
import numpy as np
from llama_cpp import Llama
import llama_cpp.llama_cpp as llama_raw
import time

MODEL_PATH = "C:/sdt_calibration/models/Mistral-7B-Instruct-v0.3-Q5_K_M.gguf"
D_MODEL = 4096

def extract_all_token_vectors(model, sentence):
    """Extract the embedding vector at every token position after eval."""
    tokens = model.tokenize(sentence.encode(), add_bos=True)
    model.reset()
    model.eval(tokens)

    vectors = {}
    for i in range(len(tokens)):
        ptr = llama_raw.llama_get_embeddings_ith(model._ctx.ctx, i)
        if ptr:
            arr = np.ctypeslib.as_array(
                ctypes.cast(ptr, ctypes.POINTER(ctypes.c_float)),
                shape=(D_MODEL,)
            ).copy()
            vectors[i] = arr
    return tokens, vectors


print("Loading Mistral on Vulkan...")
model = Llama(
    model_path=MODEL_PATH,
    n_ctx=128,
    n_gpu_layers=-1,
    verbose=False,
    embedding=True,
)

# === TEST 1: Do vectors at the same position change for different inputs? ===
print("\n=== TEST 1: Contextualisation check ===")
print("If contextualised: vectors at same position differ for different sentences.")
print("If static: vectors at same position are identical.\n")

sentences = [
    "The number 1 is a quantity.",
    "The number 9 is a quantity.",
    "The number 500 is a quantity.",
]

all_results = {}
for sent in sentences:
    tokens, vectors = extract_all_token_vectors(model, sent)
    tok_strs = [model.detokenize([t]).decode('utf-8', errors='replace') for t in tokens]
    all_results[sent] = {"tokens": tokens, "tok_strs": tok_strs, "vectors": vectors}
    print(f'"{sent}"')
    print(f"  Tokens: {tok_strs}")
    print(f"  Norms: {[f'{np.linalg.norm(vectors[i]):.1f}' for i in range(len(tokens))]}")

# Compare "The" token (position 1) across all three sentences
# If contextualised, these should differ because the rest of the sentence differs
# If static, they should be identical
print("\n--- Comparing 'The' (pos 1) across sentences ---")
v_the_1 = all_results[sentences[0]]["vectors"][1]
v_the_2 = all_results[sentences[1]]["vectors"][1]
v_the_3 = all_results[sentences[2]]["vectors"][1]

cos_12 = np.dot(v_the_1, v_the_2) / (np.linalg.norm(v_the_1) * np.linalg.norm(v_the_2))
cos_13 = np.dot(v_the_1, v_the_3) / (np.linalg.norm(v_the_1) * np.linalg.norm(v_the_3))
euc_12 = np.linalg.norm(v_the_1 - v_the_2)
euc_13 = np.linalg.norm(v_the_1 - v_the_3)

print(f"  'num 1' vs 'num 9':   cosine={cos_12:.6f}  euclidean={euc_12:.4f}")
print(f"  'num 1' vs 'num 500': cosine={cos_13:.6f}  euclidean={euc_13:.4f}")

if euc_12 < 0.001 and euc_13 < 0.001:
    print("  RESULT: STATIC embeddings (identical across inputs)")
    print("  --> Cannot use llama-cpp for Paradigm A")
    contextualised = False
else:
    print("  RESULT: CONTEXTUALISED (differ across inputs)")
    contextualised = True

# === TEST 2: Do magnitude token vectors differentiate magnitudes? ===
print("\n=== TEST 2: Magnitude differentiation ===")

# For Mistral, digits are at different positions due to tokenisation
# Let's compare the vectors at the magnitude's LAST token
carrier = "The number {N} is a quantity."
magnitudes = [1, 5, 10, 50, 100, 500, 1000]

mag_vectors = {}
for mag in magnitudes:
    sent = carrier.format(N=mag)
    tokens, vectors = extract_all_token_vectors(model, sent)
    tok_strs = [model.detokenize([t]).decode('utf-8', errors='replace') for t in tokens]

    # Find "is" position
    is_pos = None
    for i, s in enumerate(tok_strs):
        if s.strip() == "is":
            is_pos = i
            break

    last_mag_pos = is_pos - 1 if is_pos else len(tokens) - 4
    mag_vectors[mag] = vectors[last_mag_pos]
    print(f"  {mag:>5}: last_mag_pos={last_mag_pos} ('{tok_strs[last_mag_pos].strip()}')"
          f"  norm={np.linalg.norm(vectors[last_mag_pos]):.1f}")

print("\n--- Pairwise cosine similarities ---")
print(f"{'':>6}", end="")
for m2 in magnitudes:
    print(f"{m2:>8}", end="")
print()

for m1 in magnitudes:
    print(f"{m1:>5}:", end="")
    for m2 in magnitudes:
        v1 = mag_vectors[m1]
        v2 = mag_vectors[m2]
        cos = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        print(f"  {cos:.4f}", end="")
    print()

# === TEST 3: Speed benchmark ===
print("\n=== TEST 3: Extraction speed ===")
t0 = time.time()
n_bench = 26
for i, mag in enumerate([1,2,3,4,5,6,7,8,9,10,15,20,30,40,50,60,70,80,90,100,150,200,300,500,700,1000]):
    sent = carrier.format(N=mag)
    tokens, vectors = extract_all_token_vectors(model, sent)
elapsed = time.time() - t0

print(f"  26 sentences: {elapsed:.2f}s ({elapsed/26:.3f}s per sentence)")
print(f"  Paradigm A estimate (390 sentences): {elapsed/26 * 390:.1f}s")
print(f"  Three models total: {elapsed/26 * 390 * 3:.1f}s")

# === VERDICT ===
print("\n" + "=" * 60)
print("VERDICT")
print("=" * 60)
if contextualised:
    print("  llama_get_embeddings_ith returns CONTEXTUALISED vectors.")
    print("  BUT: these are FINAL LAYER ONLY.")
    print()
    print("  Option A: Use llama-cpp for everything (final-layer only)")
    print("    Pro: Fast (<20 min total), single engine, Q5_K_M throughout")
    print("    Con: Only final-layer representations, no per-layer analysis")
    print("    Impact: Cannot test H5 (layer-wise transition)")
    print()
    print("  Option B: Use HuggingFace for Paradigm A (all 33 layers)")
    print("    Pro: Full per-layer analysis, H5 testable")
    print("    Con: CPU-only FP16, ~70 min for Paradigm A, hybrid engine")
    print()
    print("  Option C: Use llama-cpp final-layer for primary, HF for H5")
    print("    Pro: Fast primary analysis + per-layer exploratory")
    print("    Con: Two extraction runs, more complexity")
else:
    print("  llama_get_embeddings_ith returns STATIC embeddings.")
    print("  Must use HuggingFace for Paradigm A.")

del model
print("\nDone.")
