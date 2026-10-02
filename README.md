# QCSA: LLM-Driven Progressive Quadruple Construction and Prompt-Augmented Learning for Robust Citation Sentiment Analysis

This repository contains the downstream training and evaluation code for QCSA. It includes the 10-fold stratified cross-validation procedure, single-backbone and multi-backbone settings, ablation experiments, and Bayesian optimization of ensemble weights.

## Repository organization

```text
QCSA/
├── README.md
├── README_CN.md
├── requirements.txt
├── train_ensemble_offline.py
│
├── athar/
│   ├── 3_w_o_NonLinearFusion.py
│   ├── 4_Baseline_roberta.py
│   └── 5_LLM_QCSA_Quad_OR_Text_Only.py
│
├── csce/
│   ├── 3_w_o_NonLinearFusion.py
│   ├── 4_Baseline_roberta.py
│   └── 5_LLM_QCSA_Quad_OR_Text_Only.py
│
├── data/
│   ├── citation_sentiment_corpus_expand_athar.csv
│   └── citation_sentiment_corpus_expand.csv
│
│
└── models/
    ├── bert-base-uncased/
    ├── roberta-base/
    └── electra-base-discriminator/
```

## Environment

The experiments reported in the paper were conducted under the following environment:

```text
Ubuntu 22.04
Python 3.10
PyTorch 2.7.1
CUDA 12.8
NVIDIA GeForce RTX 5090 (32 GB)
bf16 mixed precision
```

Install the Python dependencies with:

```bash
pip install -r requirements.txt
```

The released scripts load pretrained models from local directories. Download the required Hugging Face checkpoints in advance and place them under `models/`, or provide another local directory through `--model_dir`.

The multi-backbone setting uses:

```text
bert-base-uncased
roberta-base
electra-base-discriminator
```

## Input data

The downstream scripts read a merged JSON file containing the citation text, the gold sentiment label, and cached structured quadruples.

Each sample follows the form:

```json
{
  "text": "citation context",
  "overall_sentiment": "positive",
  "sentiment_quadruples": [
    ["aspect", "opinion", "METHODOLOGY", "positive"]
  ]
}
```

## Cached quadruples

All quadruples used in the main downstream experiments are generated offline and cached before model training. The same cached structured inputs are reused across model configurations to avoid introducing additional LLM sampling variation during downstream comparisons.

For reproducibility, the exact cached files used for the reported experiments should be released under `data/`.

## Evaluation protocol

The main evaluation uses 10-fold stratified cross-validation.

The outer folds are generated with:

```python
StratifiedKFold(
    n_splits=10,
    shuffle=True,
    random_state=41
)
```

For multi-backbone ensemble experiments, 10% of each outer training fold is further held out as an inner validation set:

```python
StratifiedShuffleSplit(
    n_splits=1,
    test_size=0.1,
    random_state=41 + fold_id
)
```

The inner validation set is used only to optimize ensemble weights. The outer test fold is never used for weight selection.

After the weights are fixed, each backbone is retrained on the complete outer training fold and evaluated only on the corresponding outer test fold.

For exact reproducibility, the fold indices used in the reported experiments should also be released under `splits/`. The sample order in the cached JSON files must remain unchanged.

## Bayesian ensemble-weight optimization

The multi-backbone QCSA setting combines:

```text
BERT-base-uncased
RoBERTa-base
ELECTRA-base-discriminator
```

The weight search uses Gaussian-process Bayesian optimization implemented with `skopt.gp_minimize`.

The objective is Macro-F1 on the inner validation set. Each model weight is searched in `[0, 1]`, with:

```text
n_calls = 50
random_state = 42
```

The resulting weights are normalized to sum to 1.

## Training settings

Unless otherwise specified, the downstream experiments use:

```text
epochs                  = 3
learning rate           = 2e-5
batch size              = 32
gradient accumulation   = 1
weight decay            = 0.05
warmup ratio            = 0.1
dropout                  = 0.1
maximum sequence length = 512
learning-rate scheduler = cosine
random seed             = 41
precision               = bf16
loss                     = standard cross-entropy
```

The released downstream scripts use a fixed three-epoch training budget and do not use early stopping.

## Reproducing the multi-backbone QCSA experiment

Example:

```bash
python train_ensemble_offline.py   --merged_json data/athar_cached_quadruples.json   --model_dir models   --seed 41   --epochs 3   --learning_rate 2e-5   --batch_size 32   --accumulation_steps 1   --weight_decay 0.05   --warmup_ratio 0.1   --bf16 1   --max_length 512
```

The script performs the following procedure:

1. construct the outer 10 stratified folds;
2. create a 10% inner validation set within each outer training fold;
3. train BERT, RoBERTa, and ELECTRA on the inner training subset;
4. obtain probability outputs on the inner validation set;
5. optimize the ensemble weights using Bayesian optimization;
6. retrain each backbone on the complete outer training fold;
7. evaluate the weighted ensemble on the outer test fold;
8. report mean Accuracy and Macro-F1 across the 10 outer folds.

## Reproducing QCSA-Single

Example with RoBERTa:

```bash
python csce/4_Baseline_roberta.py   --merged_json data/csce_cached_quadruples.json   --model_dir models   --model_name roberta-base   --seed 41   --epochs 3   --learning_rate 2e-5   --batch_size 32   --accumulation_steps 1   --weight_decay 0.05   --warmup_ratio 0.1   --bf16 1   --max_length 512
```

The single-backbone setting uses the same fixed downstream training configuration and 10-fold stratified evaluation.

## Ablation experiments

### Without nonlinear/gated fusion

```bash
python athar/3_w_o_NonLinearFusion.py   --merged_json data/athar_cached_quadruples.json   --model_dir models   --seed 41
```

### Quadruple-only

```bash
python athar/5_LLM_QCSA_Quad_OR_Text_Only.py   --merged_json data/athar_cached_quadruples.json   --model_dir models   --seed 41   --ablation_mode quad_only
```

### Text-only

```bash
python athar/5_LLM_QCSA_Quad_OR_Text_Only.py   --merged_json data/athar_cached_quadruples.json   --model_dir models   --seed 41   --ablation_mode text_only
```

Use the corresponding dataset-specific script for CSCE.

## Output

The scripts report Accuracy and Macro-F1 for each outer fold and the mean values across the 10 folds.

Training checkpoints/log directories are created by the scripts when applicable. The ensemble script also writes misclassified examples to:

```text
misclassified_samples.json
```


