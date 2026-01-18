Based on the sources, the KIVI algorithm is a **tuning-free asymmetric 2-bit quantization** method designed to compress the Key-Value (KV) cache in Large Language Models (LLMs). 

The algorithm addresses the memory bottleneck of large batch sizes and long contexts by reducing the memory footprint of the KV cache while maintaining accuracy. It relies on a specific observation regarding the element distribution of keys and values, leading to an "asymmetric" approach where keys and values are quantized differently.

Here is a detailed explanation of the algorithm's logic, structure, and execution flow.

### 1. Core Principle: Asymmetric Quantization
KIVI determines the quantization dimension based on the distinct patterns found in Key and Value matrices:
*   **Key Cache ($K$):** Quantized **per-channel**. Analysis shows that key matrices contain outlier channels with very large magnitudes. Grouping elements along the channel dimension confines quantization errors to those specific channels, preventing them from impacting normal channels.
*   **Value Cache ($V$):** Quantized **per-token**. Value matrices do not show channel-wise outliers. However, because the attention output acts as a weighted sum of value vectors, per-token quantization confines errors to individual tokens, ensuring the quantization of one token does not negatively affect others.

### 2. Data Structure: Grouped and Residual Caches
Implementing per-channel quantization in a streaming setup (where tokens arrive one by one) is difficult because the algorithm cannot see future tokens to normalize the channel. To solve this, KIVI splits the KV cache into two components:

1.  **Grouped Part ($K_g, V_g$):** This stores older tokens that have been fully quantized.
    *   **Keys:** Stored in 2-bit precision using group-wise **per-channel** quantization.
    *   **Values:** Stored in 2-bit precision using **per-token** quantization.
2.  **Residual Part ($K_r, V_r$):** This stores the most recent tokens in **full precision** (e.g., 16-bit).
    *   This acts as a "sliding window" for local relevant tokens.
    *   The size of this window is determined by a hyperparameter called the **residual length ($R$)**.

### 3. Execution Workflow (Decoding Phase)
The algorithm operates sequentially during the decoding phase. When a new token embedding $t$ arrives:

#### Step A: Update Residual Cache
First, the algorithm computes the new key ($t_K$) and value ($t_V$) vectors. These are immediately appended to the full-precision residual caches, $K_r$ and $V_r$.

#### Step B: Quantize and Flush (The Threshold Check)
The algorithm checks if the residual cache has collected enough tokens to form a complete group:
*   **For Keys:** When the length of $K_r$ reaches the residual length $R$, the entire $K_r$ tensor is quantized per-channel. This quantized block is concatenated to the grouped storage ($K_g$), and $K_r$ is reset to an empty tensor.
*   **For Values:** KIVI maintains a queue. When $V_r$ exceeds length $R$, the oldest tokens are popped, quantized per-token, and appended to the grouped storage ($V_g$).

#### Step C: Tiled Matrix Multiplication
To compute the attention scores, KIVI treats the quantized (historical) and full-precision (recent) parts separately using a tiled matrix multiplication approach:

1.  **Compute Grouped Attention ($A_g$):** The query $t_Q$ is multiplied by the dequantized historical keys $Q(K_g)^T$.
2.  **Compute Residual Attention ($A_r$):** The query $t_Q$ is multiplied by the full-precision recent keys $K_r^T$.
3.  **Concatenate:** The results are combined to form the full attention logits $A$.

$$A = \text{Concat}([A_g, A_r])$$

Finally, the attention output ($t_O$) is computed by combining the weighted sum of the grouped values and residual values.

### 4. Mathematical Formulation
For the quantized portions, KIVI uses standard round-to-nearest quantization:
$$Q(X) = \lfloor \frac{X - z_X}{s_X} \rceil$$
$$X' = Q(X) \cdot s_X + z_X$$
where $z_X$ is the zero-point and $s_X$ is the scaling factor.
*   For **Keys**, $z_X$ and $s_X$ are calculated along the channel dimension.
*   For **Values**, they are calculated along the token dimension.

### 5. Hyperparameters and Implementation
*   **Group Size ($G$):** The algorithm groups tokens (usually every 32 tokens) to calculate quantization parameters, ensuring local adaptation.
*   **Residual Length ($R$):** This defines the size of the full-precision window. A standard setting is $R=128$.
*   **System Optimization:** KIVI fuses the dequantization process with matrix multiplication (kernel "Q_MatMul") to minimize the overhead of loading quantized data and converting it back to higher precision for computation.

In summary, KIVI allows the majority of the cache to stay in 2-bit format while keeping a small, sliding window of recent tokens in full precision to accommodate the streaming nature of inference and maintain accuracy.

Based on the KIVI algorithm, computing attention requires handling the cache in two distinct parts: the **grouped part** (quantized history) and the **residual part** (full-precision recent window). The algorithm employs a "tiled matrix multiplication" strategy to compute the results separately and then combine them.

Here is the step-by-step process for computing the attention output during the decoding phase:

### 1. Compute Attention Logits (Scores)
The model computes the similarity between the current query vector ($t_Q$) and the keys. Because the keys are stored in two different formats, this calculation is split:

*   **Grouped Attention ($A_g$):** The query is multiplied by the grouped, quantized keys ($K_g$). This covers the historical context.
    $$A_g = t_Q \cdot Q(K_g)^T$$
*   **Residual Attention ($A_r$):** The query is multiplied by the residual keys ($K_r$), which are stored in full precision (e.g., FP16). This covers the most recent tokens.
    $$A_r = t_Q \cdot K_r^T$$
*   **Concatenation:** These two sets of scores are concatenated along the token dimension to form the full attention logit vector $A$.
    $$A = \text{Concat}([A_g, A_r])$$

### 2. Normalization (Softmax)
The standard Softmax function is applied to the concatenated logits $A$ to produce probability distributions across all tokens (both historical and recent).

Once normalized, the attention weights are logically split back into two segments to correspond with the values:
*   **$A_g$**: Weights corresponding to the grouped tokens (indices $0$ to end $- R$).
*   **$A_r$**: Weights corresponding to the residual tokens (last $R$ indices).

### 3. Compute Attention Output
The final output $t_O$ is the weighted sum of the value vectors. Similar to the keys, this operation is performed in two parts:

*   **Grouped Output:** The grouped attention weights are multiplied by the quantized grouped values ($V_g$).
*   **Residual Output:** The residual attention weights are multiplied by the full-precision residual values ($V_r$).

The final result is the sum of these two components:
$$t_O = A_g \cdot Q(V_g) + A_r \cdot V_r$$

### 4. Hardware Optimization (Q_MatMul)
While the mathematical formulation suggests dequantizing $Q(K_g)$ and $Q(V_g)$ before multiplication, KIVI optimizes this for hardware efficiency. It uses a custom **mixed-precision matrix multiplication kernel (Q_MatMul)**.

This kernel fuses the dequantization step with the matrix multiplication, meaning the data is dequantized "on the fly" during the computation. This avoids the memory overhead of creating large intermediate dequantized tensors and keeps the computational cores active, addressing the bottleneck typically caused by loading the KV cache.