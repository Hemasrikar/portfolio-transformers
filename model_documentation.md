# Stage 1: Portfolio Transformer Implementation Documentation

## Overview

This document provides a detailed explanation of every component in the Stage 1 notebook, which implements the embedding architecture comparison described in the dissertation proposal. The notebook constructs an encoder-only Portfolio Transformer that processes firm-level financial characteristics through four feature encoding strategies, evaluates each variant via multi-task training across three return horizons, and identifies the best-performing architecture for Stage 2 interpretability analysis.

The architecture synthesises Kelly et al. (2022) and Kisiel and Gorse (2022) into a single framework extended with sparse cross-sectional attention, Time2Vec temporal encoding, and a Gated Residual Network replacing the standard feed-forward sublayer.


## 1. Imports and Device Configuration

The opening cell imports the core dependencies required throughout the notebook. `torch` and `torch.nn` provide the deep learning framework for model construction and training. `torch.nn.functional` supplies stateless operations such as softmax and cross-entropy. `numpy` and `pandas` handle numerical computation and tabular data manipulation respectively. The `dataclass` decorator from the standard library is used to define a structured configuration object with typed fields and default values. The `Path` class from `pathlib` enables platform-independent file path handling.

Device selection proceeds by querying `torch.cuda.is_available()`, which returns `True` when a CUDA-capable GPU is detected. When a GPU is available, all tensors and model parameters are transferred to the GPU for accelerated computation. The CUDA version and device name are printed to confirm hardware compatibility with the project's computational requirements.


## 2. Configuration Dataclass

The `Config` dataclass centralises all hyperparameters and paths governing the training process. This design ensures that every experimental run is fully reproducible from a single configuration object and that variant comparisons differ only in the `encoding_variant` field while all other settings remain constant, consistent with the controlled comparison described in the proposal.

### Data Paths

The `train_path`, `val_path`, and `test_path` fields point to the preprocessed parquet files for each data split. These are declared as `Path` objects to ensure correct separator handling across operating systems.

### Model Dimensions

`d_model` (64) sets the dimensionality of the representation space into which all characteristics are encoded. Every firm token, every encoded characteristic, and every intermediate representation in the Transformer operates in this space. `n_heads` (4) determines the number of parallel attention heads in the multi-head self-attention mechanism. Each head operates in a subspace of dimension `d_model / n_heads` = 16, allowing different heads to attend to different aspects of the cross-sectional peer structure. `n_layers` (2) specifies the number of stacked Transformer encoder blocks. `d_ff` (128) sets the inner dimensionality of the Gated Residual Network that replaces the standard feed-forward sublayer. `dropout` (0.1) controls the probability of zeroing activations during training to regularise the model.

### Sparse Attention

`top_k_attention` (50) restricts each firm to attending only to its 50 most informative peers in the cross-section. This follows the explicit sparse attention mechanism of Zhao et al. (2019) and produces a peer comparison structure that is both economically interpretable and computationally tractable.

### Time2Vec

`time2vec_dim` (16) sets the output dimensionality of the Time2Vec temporal encoding. In the current implementation this is set equal to `d_model` so that the temporal encoding can be added directly to the characteristic encoding without a projection step, consistent with the additive aggregation described in Equation 2 of the proposal.

### Encoding Variant Parameters

`ple_num_bins` (16) specifies the number of quantile bins for the Piecewise Linear Encoding variant. `periodic_num_freq` (32) specifies the number of learnable sinusoidal frequencies for the Periodic Encoding variant. These are passed to the respective encoder constructors and have no effect when a different variant is selected.

### Training Hyperparameters

`learning_rate` (1e-4) and `weight_decay` (1e-5) govern the AdamW optimiser. `max_epochs` (50) caps the training duration. `patience` (7) controls early stopping: if the validation loss does not improve for 7 consecutive epochs, training halts. `grad_clip` (1.0) bounds the global gradient norm to prevent destabilising parameter updates.

### Multi-Task Horizon Weights

The three weights `lambda_3m` (0.2), `lambda_6m` (0.5), and `lambda_12m` (0.3) govern the relative contribution of each horizon to the combined loss function. The 6-month horizon receives the largest weight because this is the horizon used for portfolio construction and rebalancing, as described in Equation 9 of the proposal.

### Remaining Fields

`n_classes` (5) sets the number of quintile bins for return classification. `max_firms` (3000) caps the number of firms per cross-section to prevent GPU memory overflow when a given month has an exceptionally large universe. `batch_size` (1) is set to 1 because each "batch" is an entire monthly cross-section. `seed` (42) is set for reproducibility across all random number generators.


## 3. Column Classification

This section reads the column metadata from `train_columns.json` and programmatically identifies which columns belong to which functional category. The classification proceeds in several steps.

### Missingness Flags

Any column ending in `_miss` is identified as a binary missingness indicator. These flags were created during the preprocessing pipeline before the ranking step, ensuring that the original pattern of missing data is preserved as model input. There are 152 such flags in total, one per base characteristic.

### K1 Characteristics (Accounting-Based with Lags)

We identify K1 characteristics as those base names that have a corresponding `_lag12` column in the dataset. These are the slowly evolving accounting-based characteristics (valuation ratios, profitability measures, leverage composites) for which the proposal specifies five annual lags at months {12, 24, 36, 48, 60} alongside the current observation. There are 97 K1 characteristics, resulting in 97 × 6 = 582 K1 feature columns.

### K0 Characteristics (Current Only)

K0 characteristics are those with missingness flags but without lag counterparts. These are the rapidly changing market-based variables (volatility, beta, liquidity, momentum) for which annual lags represent stale measurements and carry no incremental predictive content over the current value. There are 55 K0 characteristics.

### Lag Structure

The list `LAG_SUFFIXES` defines the column name suffixes for each lag position: `""` (current), `"_lag12"`, `"_lag24"`, `"_lag36"`, `"_lag48"`, `"_lag60"`. The corresponding `LAG_POSITIONS` list `[0, 12, 24, 36, 48, 60]` provides the numerical lag values passed to the Time2Vec module.

### Feature Column Assembly

The code assembles `k0_feature_cols` as the ordered list of 55 K0 column names, and `k1_feature_cols` as the ordered list of 582 K1 column names (97 characteristics × 6 lag positions). The ordering within `k1_feature_cols` is structured so that all six lag positions for a given characteristic appear consecutively, which enables the dataset class to reshape this flat vector into a three-dimensional tensor of shape (N_t, 97, 6).


## 4. Cross-Sectional Dataset

The `CrossSectionalDataset` class is a PyTorch `Dataset` that organises the panel data by month for cross-sectional attention. Unlike standard datasets where each item is a single observation, here each item is an entire cross-section consisting of all firms observed in a given month. This design reflects the fact that attention in the Portfolio Transformer operates across firms within a cross-section, not across time.

### Initialisation

The constructor receives a pandas DataFrame and groups it by the `eom` (end-of-month) date column. The sorted list of unique dates defines the length of the dataset, and a dictionary maps each date to its corresponding subset of the DataFrame.

### Item Retrieval

The `__getitem__` method extracts a single monthly cross-section and constructs four tensors.

**K0 tensor** has shape (N_t, 55), where N_t is the number of firms in that month. Each entry is a rank-normalised scalar in the interval [-0.5, 0.5].

**K1 tensor** has shape (N_t, 97, 6). The flat array of 582 K1 columns is reshaped so that the second dimension indexes characteristics and the third dimension indexes lag positions. This structure allows the model to encode each characteristic-lag pair separately and apply Time2Vec to the lag dimension.

**Missingness tensor** has shape (N_t, 152). Each entry is binary (0 or 1), indicating whether the original observation was missing prior to imputation.

**Target tensors** are constructed by discretising the continuous return targets (`target_3m`, `target_6m`, `target_12m`) into quintile labels within each cross-section. We compute the 20th, 40th, 60th, and 80th percentile breakpoints from the non-missing returns in that month and assign each firm a label from 0 to 4 via `np.digitize`. Firms with missing target values receive a label of -1, which the loss function ignores. This cross-sectional discretisation ensures that label frequencies are approximately balanced within each month, consistent with the quintile-sorting approach used in the portfolio simulation.

### Firm Cap

If a monthly cross-section exceeds `max_firms`, a random subsample is drawn to fit within GPU memory constraints. This is a practical accommodation rather than a methodological choice, and the fixed seed ensures reproducibility.

### Data Loading Function

The `load_split` function reads a parquet file, verifies that all required columns are present, fills any residual NaN values in features with 0.0 (which corresponds to the cross-sectional median under rank normalisation, as established in the preprocessing pipeline), and constructs a `CrossSectionalDataset` instance.


## 5. Time2Vec Temporal Encoding

The `Time2Vec` module implements the temporal encoding of Kazemi et al. (2019) as described in Equation 3 of the proposal. It maps each lag position (a scalar such as 0, 12, 24, 36, 48, or 60) to a dense vector in R^d_model.

### Architecture

The module maintains two learnable parameter vectors: `omega` (frequencies) and `phi` (phases), both of dimension `d_model`. Given a lag position ℓ, the raw pre-activation is computed as ω_i · ℓ + φ_i for each dimension i.

### Linear and Sinusoidal Components

Following the proposal, the first component (i = 0) is left as a linear function of the lag position, capturing the monotonic decay in predictive content as the lag increases. The remaining components (i ≥ 1) are passed through a sine function, capturing cyclical factor behaviour across the business cycle.

### Application

Time2Vec is applied exclusively to K1 (accounting-based) characteristics that enter the model at multiple lag positions. K0 characteristics, which enter only at the current value, receive a learned static embedding instead. This distinction follows the proposal's argument that applying Time2Vec at lag position zero would reduce to a constant (the linear term vanishes and the sinusoidal terms collapse to phase constants), which adds parameters without meaningful temporal content.


## 6. Gated Residual Network

The `GRN` module replaces the standard two-layer feed-forward network in the Transformer encoder block, following Kisiel and Gorse (2022) and Lim et al. (2021). The module implements Equation 6 of the proposal.

### Architecture

The GRN consists of two linear layers with a Gated Linear Unit (GLU) between them, followed by a residual connection and layer normalisation.

### Forward Pass

The input `x` is first projected to a higher-dimensional space (d_ff) through `fc1` with an ELU activation. The activated output is then projected through `fc2` to produce a vector of dimension 2 × d_model, which is split into two halves: a value component and a gate component. The gate component is passed through a sigmoid function to produce values in [0, 1], and the output is the element-wise product of the value and the gate.

### Economic Motivation

The GLU gate is the critical feature. When the return signal is weak or noisy, the sigmoid gate can close toward zero, causing the GRN to default toward the identity mapping via the residual connection. When the signal is strong, the gate opens and the nonlinear transformation is applied in full. This adaptive suppression is particularly valuable in equity prediction, where the signal-to-noise ratio varies considerably across firms and market regimes.


## 7. Feature Encoding Variants

Four encoding strategies are implemented, each mapping a rank-normalised scalar in [-0.5, 0.5] to a dense vector in R^d_model. The `build_encoder` factory function returns the appropriate encoder class based on the `encoding_variant` string in the configuration.

### Variant 1: Linear Projection

The `LinearEncoder` replicates the baseline of Kelly et al. (2022). For each of the n_features characteristics, a learnable weight vector w_k in R^d_model and bias vector b_k in R^d_model map the scalar x to the vector w_k · x + b_k. The weight and bias are stored as parameters of shape (n_features, d_model), and the forward pass broadcasts across the batch dimension. This is the simplest encoding and serves as the baseline against which the other variants are compared.

### Variant 2: Per-Feature Tokenisation

The `PerFeatureTokeniser` follows Gorishniy et al. (2021). Each characteristic receives its own projection matrix W_k in R^{1 × d_model}, so the encoding is W_k · x + b_k. The distinction from Variant 1 is largely conceptual: both perform a per-feature affine mapping from R^1 to R^d_model, but the per-feature tokeniser is presented in the literature as assigning each feature its own learned embedding space, which provides a clearer conceptual framework for the subsequent attention mechanism.

### Variant 3: Piecewise Linear Encoding

The `PiecewiseLinearEncoder` implements the PLE of Gorishniy et al. (2022), as described in Equation 7 of the proposal. The input range [-0.5, 0.5] is divided into `num_bins` (default 16) equal intervals. For each bin j, the activation is the clamped linear interpolation of the input within that bin: clamp((x - t_{j-1}) / (t_j - t_{j-1}), 0, 1). The resulting bin membership vector of length `num_bins` is then projected to R^d_model through a per-feature weight matrix.

The key property of PLE is its ability to capture threshold effects that the asset pricing literature documents as central to characteristic-return relationships. For example, the value premium concentrating in the cheapest decile produces a nonlinear relationship that a linear projection cannot represent but a piecewise linear encoding handles naturally.

### Variant 4: Periodic Encoding

The `PeriodicEncoder` implements Equation 8 of the proposal. Each scalar is mapped to a vector of `num_freq` (default 32) sinusoidal components: sin(ω_j · x + φ_j), where the frequencies ω_j and phases φ_j are learnable parameters. The sinusoidal output is then projected to R^d_model through a shared linear layer.

This encoding captures cyclical patterns in characteristic-return relationships without imposing the fixed discretisation of PLE. The learnable frequencies can adapt to the data, potentially discovering periodicity in how certain characteristics predict returns across different parts of their distribution.


## 8. Sparse Multi-Head Attention

The `SparseMultiHeadAttention` module implements the cross-sectional self-attention mechanism with top-k sparsification from Zhao et al. (2019), as described in Equations 4 and 5 of the proposal.

### Multi-Head Projection

The input firm tokens (N_t, d_model) are projected into queries, keys, and values through separate linear transformations. Each is reshaped into (1, n_heads, N_t, d_k) where d_k = d_model / n_heads, allowing parallel computation across heads.

### Scaled Dot-Product Attention

Attention scores are computed as the scaled dot product of queries and keys: QK^T / sqrt(d_k). The scaling factor prevents the dot products from growing large in magnitude as d_k increases, which would push the softmax into regions of extremely small gradients.

### Top-k Sparsification

For each query (each firm), only the k largest pre-softmax scores are retained. All scores below the k-th largest are replaced with negative infinity, which drives their post-softmax attention weights to zero. This produces a sparse attention pattern where each firm attends to at most k peers.

The value of k is controlled by `top_k_attention` in the configuration. When the number of firms in the cross-section is smaller than k, no sparsification occurs and the attention is effectively dense.

### Economic Interpretation

Cross-sectional attention operates across firms within the same month rather than across characteristics or time steps. Each firm's representation is updated by aggregating information from its most informative peers, enabling the model to learn relative valuation effects and peer comparison structures. The sparsity constraint ensures that only the most relevant peers contribute, producing attention maps that are both economically interpretable and computationally efficient.


## 9. Transformer Encoder Block

The `TransformerBlock` combines sparse multi-head attention with the Gated Residual Network in a pre-norm architecture.

### Pre-Norm Design

Layer normalisation is applied before the attention and GRN sublayers rather than after them. This ordering has been shown empirically to produce more stable training gradients in deep Transformer architectures.

### Residual Connection

The output of the attention sublayer is added to the input via a residual connection before being passed to the GRN. The GRN contains its own internal residual connection and layer normalisation, as described in its documentation above.

### Attention Weight Return

The block returns both the updated firm representations and the attention weight matrix for that layer. The attention weights are retained for the interpretability analysis in Stage 2, where they will be examined for alignment with asset pricing factors and stability across market regimes.


## 10. Portfolio Transformer (Full Model)

The `PortfolioTransformer` class assembles all components into the complete architecture described in the proposal.

### Initialisation

The constructor creates separate encoder instances for K0 and K1 characteristics using the `build_encoder` factory function, a Time2Vec module for temporal encoding, a learned static embedding matrix for K0 characteristics, a linear projection for missingness flags, the specified number of Transformer encoder blocks, and three prediction heads (one per horizon).

### Firm Token Construction

The `_encode_firm_token` method implements Equation 2 of the proposal. For each firm i at month t, the firm token z_{i,t} is constructed by additive aggregation of three components.

**K0 component.** Each of the 55 K0 characteristics is passed through the feature encoder to produce a (N_t, 55, d_model) tensor. The learned static embedding e_k^(0) for each characteristic is added, and the result is summed across the 55 characteristics to produce a (N_t, d_model) vector.

**K1 component.** For each of the 6 lag positions, the 97 K1 characteristics at that lag are passed through the feature encoder to produce a (N_t, 97, d_model) tensor. The Time2Vec encoding for that lag position is computed and added to each characteristic's representation. The result is summed across all 97 characteristics, and this is repeated for all 6 lag positions with accumulation. The final K1 component is a (N_t, d_model) vector incorporating all 582 characteristic-lag pairs.

**Missingness component.** The 152-dimensional binary missingness vector is projected to R^d_model through a single linear layer.

The three components are summed to produce the firm token z_{i,t} in R^d_model. This additive aggregation preserves the model dimension without the parameter overhead of concatenation followed by projection, following the analogy with how positional and token embeddings are combined in BERT.

### Forward Pass

The firm tokens are passed through the stack of Transformer encoder blocks, where cross-sectional attention allows each firm to attend to its peers. The final representations are fed into three independent prediction heads, each consisting of layer normalisation followed by a linear projection to R^n_classes. The three output tensors (logits_3m, logits_6m, logits_12m) have shape (N_t, 5), representing the predicted quintile probabilities for each firm at each horizon.


## 11. Training Utilities

### Multi-Task Loss

The `compute_multitask_loss` function implements Equation 9 of the proposal. For each horizon h in {3, 6, 12}, it computes the cross-entropy loss between the predicted logits and the true quintile labels, weighted by the corresponding lambda_h. Firms with missing targets (label = -1) are excluded via boolean masking. The total loss is the weighted sum across all three horizons.

### Accuracy

The `compute_accuracy` function reports the fraction of firms for which the predicted quintile (the argmax of the logits) matches the true quintile label. With 5 classes, random performance is 20%.

### Rank Correlation

The `compute_rank_correlation` function provides the primary evaluation metric. Rather than using the argmax prediction, it computes an expected score for each firm as the weighted average of quintile indices under the softmax distribution. This expected score produces a continuous ranking across firms, and the Spearman rank correlation between this ranking and the true labels measures how well the model orders firms by future returns.

The Spearman correlation is computed by converting both the predicted scores and true labels to ranks, then computing the Pearson correlation of the ranks. This metric directly measures the model's ability to sort firms by future return magnitude, which is the property that matters for portfolio construction.


## 12. Training Loop

### Single Epoch

The `train_one_epoch` function iterates over monthly cross-sections in random order. For each month, it transfers the tensors to the GPU, computes the forward pass, evaluates the multi-task loss, backpropagates gradients, clips the global gradient norm to `grad_clip`, and updates parameters via AdamW. The function returns the average loss and per-horizon losses across all months.

### Evaluation

The `evaluate` function iterates over all months in a dataset without gradient computation (wrapped in `@torch.no_grad()` for memory efficiency). It computes and returns the average loss, rank correlation, and accuracy for each horizon.

### Full Training Run

The `run_training` function orchestrates the complete training procedure. It loads the training and validation splits, initialises the model and optimiser, and trains with early stopping.

**Optimiser.** AdamW is used with the configured learning rate and weight decay. AdamW decouples weight decay from the gradient update, which has been shown to improve generalisation in Transformer architectures.

**Learning rate scheduler.** `ReduceLROnPlateau` monitors the validation loss and halves the learning rate when it fails to improve for 3 consecutive epochs. This allows the model to make large initial steps and then fine-tune as it approaches convergence.

**Early stopping.** If the validation loss does not improve for `patience` (7) consecutive epochs, training halts and the best model checkpoint (by validation loss) is restored. This prevents overfitting to the training set.

**Checkpointing.** The model state dictionary is saved to disk whenever a new best validation loss is achieved. The filename includes the encoding variant name for easy identification.


## 13. Variant Comparison

The `run_all_variants` function sequentially trains all four encoding variants (linear, per-feature, PLE, periodic) using the same configuration, differing only in the `encoding_variant` field. After training each variant, it evaluates on the held-out test set and stores the training history, test metrics, and trained model. This controlled comparison isolates the contribution of the encoding layer to predictive performance, consistent with the proposal's experimental design of holding the backbone fixed while varying only the encoding.


## 14. Portfolio Simulation

The `portfolio_simulation` function implements the quintile-based portfolio construction described in Equation 10 of the proposal.

### Scoring

For each monthly cross-section, the model produces 6-month quintile logits. The expected score under the softmax distribution is computed as the weighted sum of quintile indices, producing a continuous ranking across firms.

### Portfolio Selection

Firms in the top quintile (above the 80th percentile of the expected score distribution) are selected and equally weighted. This produces a long-only portfolio that is rebalanced at each month.

### Transaction Costs

At each rebalancing date, the turnover (number of new entries plus exits) is computed. Transaction costs of 25 basis points are applied to the dollar value of all trades, as specified in the proposal.

### Performance Metrics

The `compute_portfolio_metrics` function computes cumulative return, annualised return (scaled to 12 months), annualised volatility (monthly standard deviation multiplied by sqrt(12)), Sharpe ratio (annualised return divided by annualised volatility), and maximum drawdown (the largest peak-to-trough decline in cumulative wealth). These metrics correspond directly to the benchmarks described in the proposal.


## 15. Training Diagnostics

Two plotting functions are provided for visual inspection of the training process.

`plot_training_curves` displays the training and validation loss curves on the left panel and the 6-month validation rank correlation on the right panel. Convergence is indicated by declining loss curves that track closely between training and validation (no overfitting) and a rising rank correlation.

`plot_variant_comparison` produces a three-panel bar chart comparing all four encoding variants on the test set, showing 6-month rank correlation, quintile accuracy, and multi-task loss. The best-performing variant on rank correlation is the primary candidate for Stage 2 interpretability analysis.


## 16. Model Summary and Sanity Check

The final cell instantiates the model with synthetic data to verify dimensional correctness. It creates random tensors matching the expected input shapes (100 firms, 55 K0 characteristics, 97 K1 characteristics across 6 lags, 152 missingness flags) and performs a forward pass. The output shapes are printed and verified to be (100, 5) for each horizon's logits and (4, 100, 100) for each layer's attention weights (4 heads attending across 100 firms).

All four encoding variants are instantiated and tested to confirm that they produce identical output shapes and that parameter counts vary as expected (PLE and periodic variants have more parameters due to the bin or frequency representations).


## Appendix: Tensor Shapes Summary

| Tensor | Shape | Description |
|--------|-------|-------------|
| K0 input | (N_t, 55) | Rank-normalised K0 characteristics |
| K1 input | (N_t, 97, 6) | Rank-normalised K1 characteristics across 6 lag positions |
| Missingness flags | (N_t, 152) | Binary indicators (one per base characteristic) |
| K0 encoded | (N_t, 55, d_model) | After feature encoding and static embedding |
| K1 at single lag | (N_t, 97, d_model) | After feature encoding and Time2Vec |
| Firm token z | (N_t, d_model) | After additive aggregation |
| Attention scores | (n_heads, N_t, N_t) | Pre-softmax cross-sectional attention |
| Attention weights | (n_heads, N_t, N_t) | Post-softmax sparse attention |
| Logits per horizon | (N_t, 5) | Quintile class probabilities |


## Appendix: Hyperparameter Defaults

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| d_model | 64 | Balances expressiveness with computational cost |
| n_heads | 4 | Allows diverse attention patterns in 16-dim subspaces |
| n_layers | 2 | Sufficient depth for cross-sectional comparison |
| d_ff | 128 | 2x expansion in the GRN |
| top_k | 50 | Sparse peer set size |
| dropout | 0.1 | Standard regularisation |
| learning_rate | 1e-4 | Conservative for Adam-family optimisers |
| weight_decay | 1e-5 | Light L2 regularisation |
| patience | 7 | Early stopping window |
| lambda_3m / 6m / 12m | 0.2 / 0.5 / 0.3 | Emphasises the 6-month portfolio horizon |
| n_classes | 5 | Quintile classification |
| PLE bins | 16 | Granularity of piecewise encoding |
| Periodic frequencies | 32 | Number of sinusoidal basis functions |
