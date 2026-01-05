import torch


def compute_energy(delta_r):
    """
    Calculates the L2-norm magnitude of the state change.
    Input: delta_r [d_model] or [seq_len, d_model]
    """
    return torch.norm(delta_r, p=2, dim=-1)


def compute_complexity(delta_r):
    """
    Determines circuit coordination via SVD rank.
    Input: delta_r [seq_len, d_model]
    Note: Requires a 2D matrix (sequence of updates) to find rank.
    """
    # SVD of the update matrix
    _, s, _ = torch.linalg.svd(delta_r, full_matrices=False)

    # We use a threshold relative to the largest singular value
    threshold = s.max() * 1e-3
    rank = torch.count_nonzero(s > threshold)
    return rank


def compute_innovation(delta_r, history_deltas):
    """
    Directional novelty relative to sequence/layer history.
    Input:
        delta_r: [d_model] (current update)
        history_deltas: [N, d_model] (past updates)
    """
    # Normalize vectors for cosine similarity
    delta_unit = delta_r / (torch.norm(delta_r) + 1e-9)
    history_unit = history_deltas / (
        torch.norm(history_deltas, dim=-1, keepdim=True) + 1e-9
    )

    # Cosine similarities across all history
    similarities = torch.mv(history_unit, delta_unit)

    # Innovation = 1 - max(cos(theta))
    return 1 - torch.max(similarities)


def compute_expansion(delta_r, history_deltas):
    """
    Measures update redundancy relative to the historical subspace P_t.
    Input:
        delta_r: [d_model]
        history_deltas: [N, d_model]
    """
    # 1. Create orthonormal basis for the historical subspace (P_t)
    # Using QR decomposition on the transpose of history
    Q, _ = torch.linalg.qr(history_deltas.T)

    # 2. Project delta_r onto this subspace
    # proj_P(v) = Q @ (Q.T @ v)
    projection = Q @ (Q.T @ delta_r)

    # 3. Calculate norms
    norm_delta_sq = torch.norm(delta_r) ** 2
    norm_proj_sq = torch.norm(projection) ** 2

    # Expansion = 1 - (||proj||^2 / ||delta||^2)
    return 1 - (norm_proj_sq / (norm_delta_sq + 1e-9))
