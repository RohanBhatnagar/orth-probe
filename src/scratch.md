# Residual Update Analysis

Analyzes the update $\Delta r_t = r_t - r_{t-1}$ at layer $L$ to quantify latent computation.

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