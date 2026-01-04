# Residual Update Analysis

Analyzes the update $\Delta r_t = r_t - r_{t-1}$ at layer $L$ to quantify latent computation.

We should place hooks at hook_resid_pre, hook_resid_post at each transformer block. This is redundant, since hook_resid_pre is the same as hook_resid_post of the previous block, but it will make processing simpler, allowing us to process each layer alone.

---

### 1. Expansion Factor
Measures update redundancy relative to historical subspace $P_t$.
$$Expansion = 1 - \frac{\| \text{proj}_{P_t}(\Delta r_t) \|^2}{\| \Delta r_t \|^2}$$
* **High:** Adding new basis dimensions (Information injection).
* **Low:** Refining existing representation (Redundancy).
* *$P_t$: Span of past updates (windowed or moving average).*


### 2. Complexity
Determines circuit coordination via SVD of the update.
$$Complexity = \text{rank}(\text{SVD}(\Delta r_t))$$
* **High Rank:** Multi-circuit coordination (Reasoning).
* **Low Rank:** Sparse feature firing (Recall/Lookup).

### 3. Innovation
Directional novelty relative to sequence history.
$$Innovation = 1 - \max_{i < t} \left( \text{cos}(\Delta r_t, \Delta r_i) \right)$$
* **High:** Novel logical pivot or state transition.
* **Low:** Iterative refinement or pattern repetition.

### 4. Energy
L2-norm magnitude of the state change.
$$Energy = \| \Delta r_t \|$$
* **High:** Substantial representation shift.
* **Low:** Pass-through or marginal adjustment.

---

### Update Signatures

| Mode | Expansion | Complexity | Energy |
| :--- | :--- | :--- | :--- |
| **Refinement** | Low | Low | Low |
| **Recall** | High | Low | Medium |
| **Reasoning** | High | High | High |

---

### Implementation
1. **Delta:** $\Delta r_t = r_{L,t} - r_{L-1,t}$.
2. **Rank:** SVD on $\Delta r_t$ for **Complexity**.
3. **History:** Reference $\{\Delta r_0, ..., \Delta r_{t-1}\}$ to compute **Innovation** and **Expansion**.

### Extensions:
1. Group updates and amplify certain update directions to invoke a specific mode in base models. A form of steering.
2.


### Scratch:

Qwen QWQ-32b Preview architecture:
HookedTransformer(
  (embed): Embed()
  (hook_embed): HookPoint()
  (blocks): ModuleList(
    (0-63): 64 x TransformerBlock(
      (ln1): RMSNormPre(
        (hook_scale): HookPoint()
        (hook_normalized): HookPoint()
      )
      (ln2): RMSNormPre(
        (hook_scale): HookPoint()
        (hook_normalized): HookPoint()
      )
      (attn): GroupedQueryAttention(
        (hook_k): HookPoint()
        (hook_q): HookPoint()
        (hook_v): HookPoint()
        (hook_z): HookPoint()
        (hook_attn_scores): HookPoint()
        (hook_pattern): HookPoint()
        (hook_result): HookPoint()
        (hook_rot_k): HookPoint()
        (hook_rot_q): HookPoint()
      )
      (mlp): GatedMLP(
        (hook_pre): HookPoint()
        (hook_pre_linear): HookPoint()
        (hook_post): HookPoint()
      )
      (hook_attn_in): HookPoint()
      (hook_q_input): HookPoint()
      (hook_k_input): HookPoint()
      (hook_v_input): HookPoint()
      (hook_mlp_in): HookPoint()
      (hook_attn_out): HookPoint()
      (hook_mlp_out): HookPoint()
      (hook_resid_pre): HookPoint() --> before entering block
      (hook_resid_mid): HookPoint() --> after attn
      (hook_resid_post): HookPoint() --> after MLP
    )
  )
  (ln_final): RMSNormPre(
    (hook_scale): HookPoint()
    (hook_normalized): HookPoint()
  )
  (unembed): Unembed()
)
