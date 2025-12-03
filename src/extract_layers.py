import torch
import numpy as np
import pandas as pd
from transformer_lens import HookedTransformer
from datasets import load_dataset
from tqdm import tqdm
import gc
import modal 

app = modal.App("layer-extraction")
image = (
    modal.Image.debian_slim()
    .apt_install("git")
    .pip_install_from_requirements("requirements.txt")
)

# deepseek r1 distill models not suported by transformer lens
MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct" 
LAYERS = list(range(10, 29)) # 33 total layers 
NUM_SAMPLES = 100
MAX_NEW_TOKENS = 512 

@app.function(
    image=image,
    timeout=60 * 60 * 6,
    gpu="A100",
    secrets=[modal.Secret.from_name("huggingface-secret")],
    volumes={
        "/data": modal.Volume.from_name("mats-data", create_if_missing=True)
    }
)
def extract_layers():
    import pickle
    import os
    import h5py
    import json
    
    model = HookedTransformer.from_pretrained(
        MODEL_NAME,
        device="cuda",
        dtype=torch.bfloat16,
    )
    
    datasets_to_process = {
        "triviaqa": load_dataset("mandarjoshi/trivia_qa", "rc.nocontext", split="train"),
        "gsm8k": load_dataset("openai/gsm8k", "main", split="train")
    }

    
    for dataset_name, dataset in datasets_to_process.items():
        print(f"Processing {dataset_name}...")
        os.makedirs(f"/data/{dataset_name}", exist_ok=True)
        
        indices = np.random.choice(len(dataset), NUM_SAMPLES, replace=False)
        samples = dataset.select(indices)
        
        output_path = f"/data/{dataset_name}/hidden_states.h5"
        
        with h5py.File(output_path, "w") as f:
            for i, sample in enumerate(tqdm(samples)):
                if dataset_name == "triviaqa":
                    system_msg = "You are a helpful assistant. Answer the question with just the final answer, no reasoning or citations."
                    answer_instruction = "Answer with only the final answer as a short phrase."
                elif dataset_name == "gsm8k":
                    system_msg = "You are a helpful assistant. Solve the math problem step by step and then give the final answer."
                    answer_instruction = "Show your reasoning, then clearly state the final answer on its own line."

                prompt = build_chat_prompt(sample["question"], system_msg, answer_instruction)

                residual_info = generate_and_log(model, prompt)

                sample_group = f.create_group(f"sample_{i}")
                
                sample_group.attrs["prompt"] = str(residual_info["prompt"])
                sample_group.attrs["response"] = residual_info["response"]
                sample_group.attrs["answer"] = str(sample["answer"])
                sample_group.attrs["prompt_len"] = residual_info["prompt_len"]
                sample_group.attrs["model_name"] = residual_info["model_name"]

                for layer, residuals in residual_info["resids"].items():
                    if residuals is not None:
                        sample_group.create_dataset(
                            f"layer_{layer}", 
                            data=residuals,
                            compression="gzip",
                            compression_opts=4
                        )

                del residual_info 
                if i % 5 == 0:
                    gc.collect()
                    torch.cuda.empty_cache()

        print(f"Finished processing {dataset_name}. Saved to {output_path}")
                
                
@app.local_entrypoint()
def main():
    extract_layers.remote()
    
def generate_and_log(model: HookedTransformer, prompt: str):
    if isinstance(prompt, list):
        tokens = model.tokenizer.apply_chat_template(
            prompt, 
            tokenize=True, 
            add_generation_prompt=True, 
            return_tensors="pt"
        )
        tokens = tokens.to(model.cfg.device)
    else:
        tokens = model.to_tokens(prompt)   # [batch=1, prompt_len]
    
    prompt_len = tokens.shape[1]

    out_tokens = model.generate(
        input=tokens,
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=0.7,
        stop_at_eos=True,
    )  # [batch=1, seq_len_total]

    seq_len_total = out_tokens.shape[1]

    # add hook to each layer
    hook_names = [f"blocks.{layer}.hook_resid_post" for layer in LAYERS]

    with torch.inference_mode():
        _, cache = model.run_with_cache(
            out_tokens,
            return_type="logits",
            names_filter=hook_names,
        )

    resids = {}
    for layer in LAYERS:
        resid = cache[f"blocks.{layer}.hook_resid_post"]  # [1, seq_len_total, d_model]
        arr = (
            resid[0]  # [seq_len_total, d_model]
            .detach()
            .cpu()
            .to(torch.float16)
            .numpy()
        )
        resids[layer] = arr

    response = model.to_string(out_tokens[:, prompt_len:])[0]

    return {
        "model_name": MODEL_NAME,
        "prompt": prompt,
        "response": response,
        "prompt_len": int(prompt_len),
        "resids": resids, # layer mapped to residual
    }

def build_chat_prompt(question: str, system_msg: str, answer_instruction: str):
    messages = [
        {
            "role": "system",
            "content": system_msg,
        },
        {
            "role": "user",
            "content": f"{question}\n\n{answer_instruction}",
        },
    ]
    return messages
