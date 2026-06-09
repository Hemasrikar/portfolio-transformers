# Dual Path Portfolio Transformer: Architecture and Implementation

## 1. Motivation and Design Rationale

### 1.1 Limitations of the Original Architecture

The original portfolio Transformer, following Kelly et al. (2022), represents each firm as a single token of dimension $d$ constructed by summing the encoded representations of all characteristic lag pairs:

$$z_{i,t} = \sum_{k \in K_0} h_{k,0} + \sum_{k \in K_1} \sum_{\ell \in \{0,12,24,36,48,60\}} h_{k,\ell} + z_i^{(\text{miss})}$$

This additive aggregation compresses 637 feature lag pairs (55 K0 characteristics, 97 K1 characteristics across 6 lag positions, and 152 missingness flags projected to $d$ dimensions) into a single vector of dimension $d = 64$. The compression introduces three compounding problems.

First, the raw summation assigns equal weight to every feature encoding regardless of its predictive relevance for the given firm. A momentum signal and a leverage ratio contribute identically to the firm token, even though their informational content varies across firms, sectors, and market regimes.

Second, the cross sectional attention mechanism operates across firm tokens that are each a lossy compression of the full characteristic panel. The quality of the peer comparisons the model can learn is bounded above by the quality of the firm token representation.

Third, the cross sectional attention is $O(N^2)$ in the number of firms, which forces a random truncation to `max_firms = 5000` from a universe that can reach 17,000 firms per month. The truncated subsample excludes the majority of firms from peer comparisons and introduces selection bias.

### 1.2 The Dual Path Solution

The Dual Path Transformer resolves the tension between per firm encoding quality and cross sectional comparison by separating them into two explicitly distinct computational paths.

Path 1 (the per firm encoder) processes all firms independently through attention weighted feature aggregation and a multi layer scoring head, producing a base score for every firm in the universe without any cross sectional interaction.

Path 2 (the cross sectional attention module) operates per country on the firm embeddings from Path 1, using the same sparse attention and Gated Residual Network as the original architecture, and produces a peer relative adjustment score.

The final predicted score is the sum of the two:

$$\hat{s}_i^{(h)} = f_{\text{firm}}^{(h)}(z_i) + g_{\text{cross}}^{(h)}(z_i, z_{-i}^{(c)})$$

where $f_{\text{firm}}^{(h)}$ denotes the per firm scoring head for horizon $h$, and $g_{\text{cross}}^{(h)}$ denotes the cross sectional adjustment conditioned on firm $i$'s country peer group $z_{-i}^{(c)}$.


## 2. Architecture Components

### 2.1 Feature Encoding

The encoding stage is shared across both paths and retains the controlled comparison structure of the proposal. Five encoding variants are evaluated within the same backbone:

**Variant 1: Linear Projection.** Each scalar characteristic $\tilde{x}_{i,t,k}$ is mapped to $\mathbb{R}^d$ via a per feature weight and bias: $e_k = w_k \tilde{x}_{i,t,k} + b_k$. This replicates the Kelly et al. (2022) baseline.

**Variant 2: Per Feature Tokenisation.** Each characteristic receives its own projection matrix $W_k \in \mathbb{R}^{1 \times d}$, allowing the model to learn characteristic specific transformations: $e_k = W_k \tilde{x}_{i,t,k} + b_k$. This follows Gorishniy et al. (2021).

**Variant 3: Piecewise Linear Encoding.** The rank normalised range $[-0.5, 0.5]$ is partitioned into $T$ equal width bins, and each scalar is encoded as a quantile bin membership vector with learned weights per bin per feature. This captures threshold effects such as the value premium concentrating in the cheapest decile (Gorishniy et al., 2022).

**Variant 4: Periodic Encoding.** Each scalar is mapped to a learnable sinusoidal basis: $v(\tilde{x})_j = \sin(\omega_j \tilde{x} + \phi_j)$, followed by a linear projection to $\mathbb{R}^d$. This captures cyclical characteristic return patterns without imposing a fixed discretisation (Gorishniy et al., 2022).

**Variant 5: Fourier Encoding.** Each scalar is mapped through both sine and cosine components: $[\sin(\omega_j \tilde{x}), \cos(\omega_j \tilde{x})]$, concatenated and projected to $\mathbb{R}^d$. The inclusion of both components provides a complete basis.

K0 characteristics (current period only) receive a learned static embedding $e_k^{(0)} \in \mathbb{R}^d$ added to the encoding output. K1 characteristics (with five annual lags) receive Time2Vec temporal encoding at each lag position, as described in the proposal.

### 2.2 Attention Weighted Aggregation with Missingness Masking

The raw summation is replaced by a learned attention weighted pooling mechanism. A learned query vector $q \in \mathbb{R}^d$ computes dot product scores against each encoded feature, and the softmax weighted combination produces the aggregated token.

For K0 characteristics:

$$\alpha_k^{(0)} = \text{softmax}\bigg(\frac{q_0^\top h_{k,0}}{\sqrt{d}} - \gamma \cdot m_{i,t,k}\bigg)$$

$$z_i^{(0)} = \sum_{k \in K_0} \alpha_k^{(0)} \, h_{k,0}$$

where $m_{i,t,k} \in \{0, 1\}$ is the missingness indicator for characteristic $k$, and $\gamma$ is a learned scalar penalty. When a characteristic is missing and was imputed to the cross sectional median, the penalty $\gamma \cdot 1$ is subtracted from its attention score before the softmax, driving its aggregation weight toward zero. The model therefore learns to ignore imputed feature encodings and concentrate the firm token's representational capacity on genuinely observed characteristics.

This implements the attention masking approach described in Appendix A.1.1 of the proposal, which notes that "excluding missing feature positions from the attention computation entirely by setting their pre softmax logits to $-\infty$" provides a more principled treatment of missing data than static imputation. The learned penalty $\gamma$ is a soft version of this: rather than a hard $-\infty$ mask, the model learns how aggressively to down weight missing features, which allows the possibility that the fact of missingness itself carries predictive content.

For K1 characteristics, a two level aggregation is applied. The first level pools across the 6 lag positions for each characteristic:

$$\beta_{k,\ell} = \text{softmax}\bigg(\frac{q_{\text{lag}}^\top h_{k,\ell}}{\sqrt{d}}\bigg)$$

$$\bar{h}_k = \sum_{\ell \in \{0,12,24,36,48,60\}} \beta_{k,\ell} \, h_{k,\ell}$$

The second level pools across K1 characteristics with missingness masking:

$$\alpha_k^{(1)} = \text{softmax}\bigg(\frac{q_1^\top \bar{h}_k}{\sqrt{d}} - \gamma_1 \cdot m_{i,t,k}\bigg)$$

$$z_i^{(1)} = \sum_{k \in K_1} \alpha_k^{(1)} \, \bar{h}_k$$

The firm embedding is the sum: $z_i = z_i^{(0)} + z_i^{(1)}$.

The separate `miss_proj` linear layer from the original architecture is removed. Missingness information is no longer a separate additive pathway but is integrated directly into the feature selection mechanism.

### 2.3 Path 1: Per Firm Scoring Head

The firm embedding $z_i$ passes through a multi layer perceptron with tunable depth producing a base score per horizon:

$$f_{\text{firm}}^{(h)}(z_i) = \text{MLP}^{(h)}(z_i)$$

The MLP consists of a LayerNorm followed by $L$ layers of (Linear, ELU, Dropout), and a final Linear projection to a scalar. The number of layers $L$ is a hyperparameter searched over $\{1, 2, 3\}$. With $L = 1$, the head reduces to a single hidden layer. With $L = 2$ or $L = 3$, the head has substantially more capacity to learn nonlinear characteristic return mappings.

Three separate heads are instantiated for the 3 month, 6 month, and 12 month horizons, sharing no parameters.

### 2.4 Path 2: Per Country Cross Sectional Attention

The cross sectional module takes the firm embeddings $z_i$ from the aggregation output (not from the scoring head) and runs the standard Transformer encoder blocks: pre LayerNorm, sparse multi head attention with top $k$ selection, followed by the Gated Residual Network.

The critical design choice is that cross sectional attention operates per country rather than across the full universe. For each month $t$, firms are grouped by their exchange country identifier `excntry`. Within each country group of at least `min_firms_attention` firms, the firm embeddings pass through the shared attention blocks. The attention parameters ($W_Q$, $W_K$, $W_V$, $W_O$, GRN weights) are shared across all countries, so the model learns a universal peer comparison mechanism applied to country specific cross sections.

This design resolves the `max_firms` truncation problem. The largest single country cross section (typically China or India) has roughly 3000 to 5000 firms per month, which fits within a single attention pass. All other countries have smaller cross sections. The total memory across all countries is bounded by the largest single country, and every firm in the universe participates in cross sectional attention within its own country.

A lightweight adjustment head (LayerNorm followed by a single Linear layer) maps the contextualised embedding to a scalar adjustment per horizon:

$$g_{\text{cross}}^{(h)}(z_i, z_{-i}^{(c)}) = w_h^\top \, \text{CrossSectionalBlocks}(z_i, z_{-i}^{(c)}) + b_h$$

For countries with fewer than `min_firms_attention` firms (default 10), the cross sectional path is skipped and the adjustment is zero. These firms receive only the base score from Path 1, which is trained to be independently predictive via the auxiliary loss.

### 2.5 Score Combination and Loss Function

The final predicted score for firm $i$ at horizon $h$ is:

$$\hat{s}_i^{(h)} = f_{\text{firm}}^{(h)}(z_i) + g_{\text{cross}}^{(h)}(z_i, z_{-i}^{(c)})$$

The training objective combines a main Huber loss on the combined scores with an auxiliary Huber loss on the base scores alone:

$$\mathcal{L} = \sum_{h \in \{3,6,12\}} \lambda_h \, L_{\text{Huber}}(\hat{s}^{(h)}, r^{(h)}) + \lambda_{\text{aux}} \sum_{h \in \{3,6,12\}} \lambda_h \, L_{\text{Huber}}(f_{\text{firm}}^{(h)}(z), r^{(h)})$$

where $r^{(h)}$ denotes the realised cumulative excess return at horizon $h$, and $\lambda_{\text{aux}}$ is a tunable hyperparameter controlling the strength of the auxiliary regularisation.

The auxiliary loss serves two purposes. First, it ensures that the per firm encoder learns to produce meaningful scores independently, so that firms excluded from the cross sectional subsample during inference still receive a well calibrated prediction. Second, it regularises the combined model by preventing the cross sectional path from compensating for a weak per firm encoder.

The Huber loss is used with $\delta = 1.0$, which provides robustness to extreme returns without the training instability observed with ranking losses (ListNet) on this dataset.


## 3. Interpretability Outputs

The architecture produces three layers of interpretability without any post hoc analysis.

### 3.1 Feature Importance Weights

The K0 aggregation weights $\alpha_k^{(0)}$ and the K1 feature aggregation weights $\alpha_k^{(1)}$ are firm specific and month specific. They indicate which characteristics the model considers most important for each firm at each point in time. Averaging across firms within a month gives a monthly feature importance ranking. Conditioning on market regimes (bull, bear, high volatility) reveals whether the model's feature attention shifts in economically coherent ways. These weights are returned from the forward pass and can be extracted at inference without additional computation.

### 3.2 Temporal Lag Importance

The K1 lag aggregation weights $\beta_{k,\ell}$ show, for each K1 characteristic, how much weight the model places on the current observation versus historical annual lags. If the model attends primarily to lag 0 (the current value), the historical lags are not contributing incremental signal. If it attends to lag 48 or lag 60, the model is detecting long term mean reversion or multi year cyclical patterns.

### 3.3 Cross Sectional Peer Structure

The sparse attention weights from Path 2 reveal which firms within each country the model compares against when forming the peer relative adjustment. These attention patterns can be analysed for alignment with industry structure, market capitalisation groupings, or other economically meaningful firm taxonomies.


## 4. Code Structure

### 4.1 Configuration

The `Config` dataclass centralises all hyperparameters. Key additions relative to the original architecture:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_mlp_layers` | 2 | Depth of the per firm scoring MLP (search: 1 to 3) |
| `lambda_aux` | 0.3 | Auxiliary loss weight on base scores (search: 0.1 to 0.5) |
| `min_firms_attention` | 10 | Minimum country size to activate Path 2 |
| `raw_path` | `../results/Global Factor_EM.parquet` | Path to raw data for country lookup |

### 4.2 Dataset

The `CrossSectionalDataset` stores per month tensors with the following structure:

| Key | Shape | Description |
|-----|-------|-------------|
| `k0` | `(N, n_k0)` | Rank normalised K0 characteristics |
| `k1` | `(N, n_k1, 6)` | Rank normalised K1 characteristics across 6 lags |
| `k0_miss` | `(N, n_k0)` | Binary missingness flags for K0 |
| `k1_miss` | `(N, n_k1)` | Binary missingness flags for K1 |
| `country_ids` | `(N,)` | Integer country identifiers |
| `targets` | dict of `(N,)` | Continuous returns per horizon |
| `valid_masks` | dict of `(N,)` | Boolean validity per horizon |

Country identifiers are built from a lookup table `COUNTRY_LOOKUP` constructed by reading `(id, eom, excntry)` from the raw parquet and mapping each country string to a stable integer via `COUNTRY_TO_ID`. The `eom` column is explicitly converted to `datetime64` before the merge to prevent dtype mismatches with the processed parquet files.

### 4.3 Model Classes

**`AttentiveAggregation`**: Attention weighted feature pooling for K0 characteristics. Takes encoded features of shape `(N, n_features, d)` and an optional missingness mask of shape `(N, n_features)`. Returns the aggregated token `(N, d)` and the attention weights `(N, n_features)`.

**`K1TwoLevelAggregation`**: Two level attention pooling for K1 characteristics. First pools across lags (dim 1 of shape `(N, 6, n_k1, d)`), then pools across features. Returns the aggregated token, lag weights `(N, 6, n_k1)`, and feature weights `(N, n_k1)`.

**`FirmScoreHead`**: Variable depth MLP for base score prediction. Constructed from a LayerNorm, $L$ hidden layers (Linear, ELU, Dropout), and a final Linear to scalar. The depth $L$ is set by `config.n_mlp_layers`.

**`DualPathTransformer`**: The main model class. The `_encode_firms` method runs the shared encoding and aggregation (producing firm embeddings for all firms). The `forward` method runs both paths: base scores from `FirmScoreHead` applied to all firms, and adjustment scores from per country cross sectional attention. Returns a dictionary containing combined scores, base scores, attention weights, and aggregation weights.

### 4.4 Training Loop

Early stopping and the learning rate scheduler both monitor validation 6 month rank correlation (`mode = "max"`), not validation loss. This addresses the proxy mismatch where a model can reduce Huber loss by improving predictions for the middle of the return distribution while the tails (which drive portfolio performance) stagnate.

The `GradScaler` is instantiated once per training run for mixed precision training. Gradient clipping at `config.grad_clip` prevents exploding gradients from extreme returns.

### 4.5 Portfolio Simulation

The portfolio simulation uses the combined score (`output["scores_6m"]`) for quintile selection. The top quintile is equally weighted for the long only portfolio and score proportionally weighted for the long short portfolio. Transaction costs of 25 basis points are applied based on portfolio turnover at each 6 month rebalancing date.


## 5. Hyperparameter Tuning

The tuning script (`embeding_varients.py`) runs independent Optuna studies per encoding variant, each maximising validation long short Sharpe ratio with 6 month rebalancing and quintile (20%) selection.

The search space comprises 15 parameters:

| Category | Parameters |
|----------|-----------|
| Architecture | `d_model` [64, 96, 128], `n_heads` [2, 4, 8], `d_ff_mult` [2, 4], `n_layers` [1, 3], `dropout` [0.01, 0.4], `top_k` [10, 20, 50, 100] |
| Dual path | `n_mlp_layers` [1, 3], `lambda_aux` [0.1, 0.5] |
| Optimiser | `lr` [5e-5, 5e-3], `weight_decay` [1e-7, 1e-2], `grad_clip` [0.1, 5.0] |
| Multi task | `lambda_3m` [0.05, 0.45], `lambda_12m` [0.05, 0.45] |
| Encoding | `ple_num_bins` [8, 16, 32], `periodic_num_freq` [16, 32, 64] |

Trial persistence uses SQLite (`hpt_dual_path.db`), enabling kernel restarts without loss of completed trials. The `MedianPruner` terminates unpromising trials after 6 warmup epochs, and `n_jobs = 1` prevents threading race conditions.


## 6. Comparison with the Original Architecture

| Aspect | Original (Kelly et al. based) | Dual Path |
|--------|-------------------------------|-----------|
| Firm token construction | Raw sum of 637 feature encodings | Attention weighted aggregation with missingness masking |
| Missing data handling | Separate linear projection added to firm token | Integrated into aggregation attention scores |
| Per firm scoring | Single linear layer | Multi layer MLP with tunable depth |
| Cross sectional scope | All firms (truncated to `max_firms`) | Per country (no truncation required) |
| Scoring decomposition | Single combined score | Base score + cross sectional adjustment |
| Training objective | Single Huber loss | Main + auxiliary Huber loss |
| Early stopping criterion | Validation loss | Validation 6 month rank correlation |
| Interpretability | Post hoc (SHAP, attention rollout) | Native (aggregation weights, lag weights, attention) |
| Firms scored at inference | Only `max_firms` subsample | All firms in universe |
