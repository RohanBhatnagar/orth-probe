import torch
import numpy as np
from transformer_lens import HookedTransformer
from tqdm import tqdm
import modal
from collections import defaultdict
import sys
from pathlib import Path
import hydra
from config_runtime import init_cfg
import json
from metric_functions import (
    compute_energy,
    compute_complexity,
    compute_innovation,
    compute_expansion,
)

CONFIG_DIR = str(Path(__file__).parent / "configs")

# Add util directory to path for imports
sys.path.append(str(Path(__file__).parent / "util"))


app = modal.App("compute-metrics")

image = (
    modal.Image.debian_slim()
    .apt_install("git")
    .pip_install_from_requirements("requirements.txt")
    .add_local_dir(".", remote_path="/root")
)


@app.function(
    image=image,
    volumes={"/data": modal.Volume.from_name("orth-data", create_if_missing=True)},
    secrets=[modal.Secret.from_name("huggingface-secret")],
    gpu="A100",
    timeout=60 * 60 * 6,
)
def compute_metrics_main(cfg):
    """
    Load residual data and compute the four metrics:
    1. Energy: L2-norm magnitude of state change
    2. Complexity: Rank via SVD
    3. Innovation: Directional novelty relative to history
    4. Expansion: Update redundancy relative to historical subspace
    """
    import h5py
    import os
    import sys
    from omegaconf import OmegaConf

    cfg = OmegaConf.create(cfg)

    # Add util to path in remote environment
    sys.path.append("/root/util")

    print(f"Loading model: {cfg.model.name}")
    model = HookedTransformer.from_pretrained(
        cfg.model.name,
        device=cfg.model.device,
        dtype=cfg.model.dtype,
    )

    # Construct path to residuals file
    dataset_alias = cfg.dataset.path.split("/")[-1]
    residuals_path = f"/data/{cfg.model.name}/{dataset_alias}/residuals.h5"

    if not os.path.exists(residuals_path):
        print(f"Error: Residuals file not found at {residuals_path}")
        return

    print(f"Loading residuals from {residuals_path}")

    results = defaultdict(lambda: defaultdict(list))

    with h5py.File(residuals_path, "r") as f:
        sample_keys = sorted(list(f.keys()), key=lambda x: int(x.split("_")[1]))

        for sample_key in tqdm(sample_keys, desc="Processing samples"):
            grp = f[sample_key]

            # Get metadata
            # prompt = grp.attrs.get("prompt", "")
            # response = grp.attrs.get("response", "")
            # prompt_len = grp.attrs.get("prompt_len", 0)

            # Load residuals for all layers
            # Structure: layer_L_pre and layer_L_post for each layer L
            layers_data = {}

            for dset_name in grp.keys():
                if dset_name.startswith("layer_"):
                    parts = dset_name.split("_")
                    layer_idx = int(parts[1])
                    residual_type = parts[2]  # 'pre' or 'post'

                    if layer_idx not in layers_data:
                        layers_data[layer_idx] = {}

                    # Load as torch tensor
                    layers_data[layer_idx][residual_type] = (
                        torch.from_numpy(grp[dset_name][:]).to(torch.float32).cuda()
                    )

            # Process each layer
            for layer_idx in sorted(layers_data.keys()):
                if (
                    "pre" not in layers_data[layer_idx]
                    or "post" not in layers_data[layer_idx]
                ):
                    continue

                resid_pre = layers_data[layer_idx]["pre"]  # [seq_len, d_model]
                resid_post = layers_data[layer_idx]["post"]  # [seq_len, d_model]

                # Compute delta: change across this layer
                delta_r = resid_post - resid_pre  # [seq_len, d_model]

                seq_len = delta_r.shape[0]

                # Compute metrics for each token position
                for t in range(seq_len):
                    delta_t = delta_r[t]  # [d_model]

                    # 1. Energy: L2-norm of the update
                    energy = compute_energy(delta_t).item()

                    # 2. Complexity: requires multiple updates, use window
                    # We'll compute complexity over a sliding window
                    if t >= cfg.history_window:
                        delta_window = delta_r[
                            t - cfg.history_window : t + 1
                        ]  # [window_size, d_model]
                        complexity = compute_complexity(delta_window).item()
                    else:
                        complexity = None

                    # 3. Innovation: novelty relative to past updates
                    if t > 0:
                        history_deltas = delta_r[:t]  # [t, d_model]
                        innovation = compute_innovation(delta_t, history_deltas).item()
                    else:
                        innovation = None

                    # 4. Expansion: redundancy relative to historical subspace
                    if t > 0:
                        history_deltas = delta_r[:t]  # [t, d_model]
                        expansion = compute_expansion(delta_t, history_deltas).item()
                    else:
                        expansion = None

                    # Store results
                    results[sample_key][layer_idx].append(
                        {
                            "token_pos": t,
                            "energy": energy,
                            "complexity": complexity,
                            "innovation": innovation,
                            "expansion": expansion,
                        }
                    )

            # Clean up GPU memory
            del layers_data
            torch.cuda.empty_cache()

    # Analyze and visualize results
    analyze_metrics(results, model, residuals_path)

    return results


@app.local_entrypoint()
def main():
    _clean_args_for_hydra()
    _hydra_main()


def _clean_args_for_hydra():
    if "--" in sys.argv:
        sys.argv = [sys.argv[0]] + sys.argv[sys.argv.index("--") + 1 :]
    else:
        sys.argv = [sys.argv[0]]


@hydra.main(
    version_base=None, config_path=CONFIG_DIR, config_name="compute_metrics_config"
)
def _hydra_main(cfg):
    init_cfg(cfg)
    set_seed(cfg.seed)

    from omegaconf import OmegaConf

    cfg_dict = OmegaConf.to_container(cfg, resolve=True)

    print(f"Running with config:\n{json.dumps(cfg_dict, indent=4)}")

    compute_metrics_main.remote(cfg_dict)


def analyze_metrics(results, model, residuals_path):
    """
    Analyze and print statistics about the computed metrics.
    """
    import h5py

    print("\n" + "=" * 80)
    print("METRIC ANALYSIS")
    print("=" * 80)

    # Aggregate metrics across all samples and layers
    all_energy = []
    all_complexity = []
    all_innovation = []
    all_expansion = []

    for sample_key, layer_dict in results.items():
        for layer_idx, metrics_list in layer_dict.items():
            for metrics in metrics_list:
                all_energy.append(metrics["energy"])
                if metrics["complexity"] is not None:
                    all_complexity.append(metrics["complexity"])
                if metrics["innovation"] is not None:
                    all_innovation.append(metrics["innovation"])
                if metrics["expansion"] is not None:
                    all_expansion.append(metrics["expansion"])

    # Convert to numpy for statistics
    all_energy = np.array(all_energy)
    all_complexity = np.array(all_complexity)
    all_innovation = np.array(all_innovation)
    all_expansion = np.array(all_expansion)

    print("\nEnergy Statistics:")
    print(f"  Mean: {all_energy.mean():.4f}")
    print(f"  Std:  {all_energy.std():.4f}")
    print(f"  Min:  {all_energy.min():.4f}")
    print(f"  Max:  {all_energy.max():.4f}")

    print("\nComplexity Statistics:")
    print(f"  Mean: {all_complexity.mean():.4f}")
    print(f"  Std:  {all_complexity.std():.4f}")
    print(f"  Min:  {all_complexity.min():.4f}")
    print(f"  Max:  {all_complexity.max():.4f}")

    print("\nInnovation Statistics:")
    print(f"  Mean: {all_innovation.mean():.4f}")
    print(f"  Std:  {all_innovation.std():.4f}")
    print(f"  Min:  {all_innovation.min():.4f}")
    print(f"  Max:  {all_innovation.max():.4f}")

    print("\nExpansion Statistics:")
    print(f"  Mean: {all_expansion.mean():.4f}")
    print(f"  Std:  {all_expansion.std():.4f}")
    print(f"  Min:  {all_expansion.min():.4f}")
    print(f"  Max:  {all_expansion.max():.4f}")

    # Print sample-level analysis
    print("\n" + "=" * 80)
    print("SAMPLE-LEVEL ANALYSIS (First 3 samples)")
    print("=" * 80)

    with h5py.File(residuals_path, "r") as f:
        for i, (sample_key, layer_dict) in enumerate(list(results.items())[:3]):
            grp = f[sample_key]
            response = grp.attrs.get("response", "")

            print(f"\n--- Sample {sample_key} ---")
            print(f"Response: {response[:100]}...")

            # Show metrics for a few layers
            for layer_idx in sorted(layer_dict.keys())[:3]:
                metrics_list = layer_dict[layer_idx]

                # Compute layer-level averages
                layer_energy = np.mean([m["energy"] for m in metrics_list])
                layer_complexity = np.mean(
                    [
                        m["complexity"]
                        for m in metrics_list
                        if m["complexity"] is not None
                    ]
                )
                layer_innovation = np.mean(
                    [
                        m["innovation"]
                        for m in metrics_list
                        if m["innovation"] is not None
                    ]
                )
                layer_expansion = np.mean(
                    [m["expansion"] for m in metrics_list if m["expansion"] is not None]
                )

                print(f"\n  Layer {layer_idx}:")
                print(f"    Energy:     {layer_energy:.4f}")
                print(f"    Complexity: {layer_complexity:.4f}")
                print(f"    Innovation: {layer_innovation:.4f}")
                print(f"    Expansion:  {layer_expansion:.4f}")

    print("\n" + "=" * 80)


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == "__main__":
    main()
