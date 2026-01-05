import torch
import numpy as np
from transformer_lens import HookedTransformer
from datasets import load_dataset
from tqdm import tqdm
import gc
import modal
import hydra
from config_runtime import init_cfg, cfg
from pathlib import Path
import sys
import json

CONFIG_DIR = str(Path(__file__).parent / "configs")

app = modal.App("layer-extraction")
image = (
    modal.Image.debian_slim()
    .apt_install("git")
    .pip_install_from_requirements("requirements.txt")
    .add_local_dir(".", remote_path="/root")
)


@app.function(
    image=image,
    timeout=60 * 60 * 6,
    gpu="A100-80GB",
    secrets=[modal.Secret.from_name("huggingface-secret")],
    volumes={"/data": modal.Volume.from_name("orth-data", create_if_missing=True)},
)
def extract_residuals(cfg):
    import os
    import h5py
    from omegaconf import OmegaConf

    cfg = OmegaConf.create(cfg)

    model = HookedTransformer.from_pretrained(
        cfg.model.name,
        device=cfg.model.device,
        dtype=cfg.model.dtype,
    )

    dataset = load_dataset(cfg.dataset.path, cfg.dataset.name, split=cfg.dataset.split)
    os.makedirs(f"/data/{cfg.model.name}", exist_ok=True)

    dataset_alias = cfg.dataset.path.split("/")[-1]
    os.makedirs(f"/data/{cfg.model.name}/{dataset_alias}", exist_ok=True)

    print(f"Processing {dataset_alias}...")

    indices = np.random.choice(len(dataset), cfg.num_samples, replace=False)
    samples = dataset.select(indices)

    output_path = f"/data/{cfg.model.name}/{dataset_alias}/residuals.h5"

    with h5py.File(output_path, "w") as f:
        for i, sample in enumerate(tqdm(samples)):
            system_msg = cfg.dataset.system_msg
            answer_instruction = cfg.dataset.answer_instruction
            prompt = (
                system_msg + "\n\n" + sample["question"] + "\n\n" + answer_instruction
            )

            residual_info = generate_and_log(model, prompt)

            sample_group = f.create_group(f"sample_{i}")

            sample_group.attrs["prompt"] = str(residual_info["prompt"])
            sample_group.attrs["response"] = residual_info["response"]
            sample_group.attrs["answer"] = str(sample["answer"])
            sample_group.attrs["prompt_len"] = residual_info["prompt_len"]
            sample_group.attrs["model_name"] = residual_info["model_name"]

            for layer, residuals_dict in residual_info["resids"].items():
                if residuals_dict is not None:
                    if "pre" in residuals_dict:
                        sample_group.create_dataset(
                            f"layer_{layer}_pre",
                            data=residuals_dict["pre"],
                            compression="gzip",
                            compression_opts=4,
                        )
                    if "post" in residuals_dict:
                        sample_group.create_dataset(
                            f"layer_{layer}_post",
                            data=residuals_dict["post"],
                            compression="gzip",
                            compression_opts=4,
                        )

            del residual_info
            if i % 5 == 0:
                gc.collect()
                torch.cuda.empty_cache()

    print(f"Finished processing {dataset_alias}. Saved to {output_path}")


@app.local_entrypoint()
def main():
    _clean_args_for_hydra()
    _hydra_main()


def _clean_args_for_hydra():
    if "--" in sys.argv:
        sys.argv = [sys.argv[0]] + sys.argv[sys.argv.index("--") + 1 :]
    else:
        sys.argv = [sys.argv[0]]


@hydra.main(version_base=None, config_path=CONFIG_DIR, config_name="extract_resid_config")
def _hydra_main(cfg):
    init_cfg(cfg)
    set_seed(cfg.seed)

    from omegaconf import OmegaConf

    cfg_dict = OmegaConf.to_container(cfg, resolve=True)

    print(f"Running with config:\n{json.dumps(cfg_dict, indent=4)}")

    extract_residuals.remote(cfg_dict)


def generate_and_log(model: HookedTransformer, prompt: str):
    if isinstance(prompt, list):
        tokens = model.tokenizer.apply_chat_template(
            prompt, tokenize=True, add_generation_prompt=True, return_tensors="pt"
        )
        tokens = tokens.to(model.cfg.device)
    else:
        tokens = model.to_tokens(prompt)  # [batch=1, prompt_len]

    prompt_len = tokens.shape[1]

    out_tokens = model.generate(
        input=tokens,
        max_new_tokens=cfg.model.generation.max_new_tokens,
        temperature=cfg.model.generation.temperature,
        stop_at_eos=cfg.model.generation.stop_at_eos,
    )  # [batch=1, seq_len_total]

    hook_names = []
    for layer in range(len(model.blocks)):
        hook_names.append(f"blocks.{layer}.hook_resid_pre")
        hook_names.append(f"blocks.{layer}.hook_resid_post")

    with torch.inference_mode():
        _, cache = model.run_with_cache(
            out_tokens,
            return_type="logits",
            names_filter=hook_names,
        )

    residuals = {}
    for layer in range(len(model.blocks)):
        residuals[layer] = {}

        residual_pre = cache[
            f"blocks.{layer}.hook_resid_pre"
        ]  # [1, seq_len_total, d_model]
        arr_pre = (
            residual_pre[0]  # [seq_len_total, d_model]
            .detach()
            .cpu()
            .to(torch.float16)
            .numpy()
        )
        residuals[layer]["pre"] = arr_pre

        residual_post = cache[
            f"blocks.{layer}.hook_resid_post"
        ]  # [1, seq_len_total, d_model]
        arr_post = (
            residual_post[0]  # [seq_len_total, d_model]
            .detach()
            .cpu()
            .to(torch.float16)
            .numpy()
        )
        residuals[layer]["post"] = arr_post

    response = model.to_string(out_tokens[:, prompt_len:])[0]
    print(response)

    return {
        "model_name": cfg.model.name,
        "prompt": prompt,
        "response": response,
        "prompt_len": int(prompt_len),
        "resids": residuals,  # layer mapped to residual
    }


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == "__main__":
    main()
