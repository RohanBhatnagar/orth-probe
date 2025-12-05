import torch 
import numpy as np
import pandas as pd
from transformer_lens import HookedTransformer
from datasets import load_dataset
from tqdm import tqdm
import gc 
import modal 
from collections import defaultdict

app = modal.App("orthogonality-probe")

image = (
    modal.Image.debian_slim()
    .apt_install("git")
    .pip_install_from_requirements("requirements.txt")
)

MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"
LAYERS = list(range(20, 29)) # 33 total layers 

@app.function(
    image=image,
    volumes={
        '/data': modal.Volume.from_name("mats-data", create_if_missing=True)
    },
    secrets=[modal.Secret.from_name("huggingface-secret")],
    gpu="A100",
    timeout=60 * 60,
)
def orth_probe(step_size: int = 1):
    import h5py
    import os
    import pickle

    def cos_sim(a, b):
        # a: [dim], b: [dim]
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return np.dot(a, b) / (norm_a * norm_b)

    print("Loading model.")
    model = HookedTransformer.from_pretrained(
        MODEL_NAME,
        device="cuda",
        dtype=torch.bfloat16,
    )
        
    datasets = ["triviaqa", "gsm8k"]
    results = defaultdict(lambda: defaultdict(dict))

    for dataset_name in datasets:
        h5_path = f"/data/{dataset_name}/hidden_states.h5"
        if not os.path.exists(h5_path):
            print(f"Skipping {dataset_name}, file not found.")
            continue
            
        print(f"Processing {dataset_name}...")

        # sample --> tokenized response
        tokenized_responses = defaultdict(list)
        # sample --> token --> layer pair --> (cos sim)
        sims = defaultdict(lambda: defaultdict(dict))

        with h5py.File(h5_path, "r") as f:
            sample_keys = sorted(list(f.keys()), key=lambda x: int(x.split('_')[1]))
            for key in tqdm(sample_keys):
                grp = f[key]
                prompt = grp.attrs["prompt"]
                response = grp.attrs["response"]

                tokenized_response = model.to_tokens(response)
                tokenized_responses[key] = tokenized_response

                prompt_len = grp.attrs["prompt_len"]
                
                # residuals: layer --> [seq_len, dim]
                residuals = {}
                for name, dset in grp.items(): 
                    if name.startswith("layer_"):
                        layer_idx = int(name.split("_")[1])
                        residuals[layer_idx] = dset[:]

                layers = sorted(residuals.keys())
                seq_len = residuals[layers[0]].shape[0] 

                start, end = min(layers), max(layers)
                block_update = residuals[end] - residuals[start]
                norms = np.linalg.norm(block_update, axis=1, keepdims=True)
                normalized_block_update = block_update / (norms + 1e-9)

                cos_sims = np.sum(
                    normalized_block_update[:-step_size] * normalized_block_update[step_size:], axis=1
                )
                
                for t in range(len(cos_sims)):
                    sims[key][t + step_size][(start, end)] = cos_sims[t]
        
        results[dataset_name]["tokenized_responses"] = tokenized_responses
        results[dataset_name]["sims"] = sims
    
    analyze_orthogonality(results, datasets, model)

    return 

@app.local_entrypoint()
def main(step_size: int = 1):
    orth_probe.remote(step_size)

def analyze_orthogonality(
    results,
    datasets,
    model,
    max_samples=50,
):    
    for dataset in datasets:
        print(f"\n===== Analyzing {dataset} =====")
        tokenized_responses = results[dataset]["tokenized_responses"]
        sims = results[dataset]["sims"]

        n_printed = 0

        for sample_key, per_token_dict in sims.items():
            if max_samples is not None and n_printed >= max_samples:
                break

            # --- decode tokens for this sample ---
            token_tensor = tokenized_responses[sample_key]
            # Expect [1, seq_len] or [seq_len]
            if hasattr(token_tensor, "detach"):  # torch tensor
                token_ids = token_tensor.detach().cpu().numpy()
            else:
                token_ids = np.array(token_tensor)

            if token_ids.ndim == 2:
                token_ids = token_ids[0]

            # Convert token ids -> string tokens
            str_tokens = model.to_str_tokens(token_ids)

            # --- compute a score per token: min |cos| across layer pairs ---
            token_scores = {}
            for token_idx, pair_dict in per_token_dict.items():
                if not pair_dict:
                    continue
                vals = list(pair_dict.values())
                min_val = min(vals)
                max_val = max(vals)
                mean_val = sum(vals) / len(vals)
                min_abs_cos = min(abs(v) for v in vals)

                token_scores[int(token_idx)] = {
                    "min_val": min_val,
                    "max_val": max_val,
                    "mean_val": mean_val,
                    "min_abs_cos": min_abs_cos,
                }

            highlighted = ""
            for i, tok in enumerate(str_tokens):
                score_data = token_scores.get(i)
                if score_data: 
                    color = get_heatmap_ansi(score_data["min_abs_cos"])
                    highlighted += f"{color}{tok}\033[0m"

            print(f"\n===== Sample {sample_key} =====")
            print(highlighted)
            print("=================================")

            n_printed += 1

def get_heatmap_ansi(cos_sim):
    """
    Maps cosine similarity to a Red background intensity.
    Low Cos Sim (Orthogonal) -> Bright Red Background
    High Cos Sim (Parallel)  -> No Background (or Black)
    """
    intensity = 1.0 - cos_sim
    intensity = intensity ** 3 
    r = int(255 * intensity)
    g = 0 
    b = 0
    if r < 20:
        return "\033[0m"
    return f"\033[48;2;{r};{g};{b}m"