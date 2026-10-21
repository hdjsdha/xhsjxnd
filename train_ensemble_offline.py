# -*- coding: utf-8 -*-
"""
"""

import argparse
import json
import os
import random
from collections import Counter
from pathlib import Path
from typing import Optional, Dict, List, Any, Tuple
import gc
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

from sklearn.metrics import f1_score
from torch import nn
from torch.utils.data import DataLoader
from skopt import gp_minimize
from skopt.space import Real
# ✅ force transformers/huggingface offline (no download)
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

from transformers import (
    AutoConfig,
    PreTrainedModel,
    AutoModel,
    Trainer,
    PretrainedConfig,
    AutoTokenizer,
    TrainingArguments,
    default_data_collator,
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--model_dir", type=str, default="models")
    parser.add_argument("--model_name", type=str, default="roberta-base")

    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--accumulation_steps", type=int, default=1)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--bf16", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    # ✅ merged dataset ONLY
    parser.add_argument("--merged_json", type=str, default="data/dataset_merged_posneg_style.json",
                        help="Unified dataset produced by build_dataset.py")
    # settings
    parser.add_argument("--max_length", type=int, default=512)
    args = parser.parse_args()

    print("⚙️ [INFO] Training configuration:")
    print(f"   Merged dataset: {args.merged_json}")
    print(f"   Model dir: {args.model_dir}")
    print(f"   Epochs: {args.epochs}")
    print(f"   LR: {args.learning_rate}")
    print(f"   Batch size: {args.batch_size}")
    print(f"   Device: {args.device}")

    seed_everything(args.seed)
    main(args)
