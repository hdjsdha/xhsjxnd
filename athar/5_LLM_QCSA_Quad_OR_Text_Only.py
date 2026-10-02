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

# -------------------------
# Optional local imports (keep if you have them)
# -------------------------
try:
    from bert.plot_results import plot_confusion_matrix  # noqa: F401
except Exception:
    plot_confusion_matrix = None


# =========================
# Path Configuration
# =========================
PROJECT_ROOT = Path(__file__).resolve().parent
print(f"🔧 [INFO] Project root directory: {PROJECT_ROOT}")
print(f"🔧 [INFO] Current working directory: {os.getcwd()}")

def clean_memory():
    torch.cuda.empty_cache()  # 清理GPU显存
    gc.collect()
# =========================
# Reproducibility
# =========================
def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def masked_mean_pool(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.float()
    denom = mask.sum(dim=1, keepdim=True).clamp(min=1.0)
    pooled = (hidden * mask.unsqueeze(-1)).sum(dim=1) / denom
    return pooled


# =========================
# Local Model Path Resolver
# =========================
def resolve_local_model_path(model_dir: str, model_name: str) -> str:
    """
    Only load local model.
    Search:
      1) PROJECT_ROOT/models/<model_name>
      2) <model_dir>/<model_name> (relative to PROJECT_ROOT if not abs)
      3) PROJECT_ROOT/pretrain_models/<model_name>
      4) PROJECT_ROOT.parent/models/<model_name>
    """
    print(f"\n📂 [DEBUG] Looking for local model: {model_name}")
    print(f"📂 [DEBUG] Model dir argument: {model_dir}")

    direct_path = PROJECT_ROOT / "models" / model_name
    if direct_path.exists():
        print(f"✅ [SUCCESS] Found model at project models: {direct_path}")
        return str(direct_path)

    if model_dir:
        user_path = Path(model_dir)
        if not user_path.is_absolute():
            user_path = PROJECT_ROOT / user_path
        model_path = user_path / model_name
        if model_path.exists():
            print(f"✅ [SUCCESS] Found model at user path: {model_path}")
            return str(model_path)
        else:
            print(f"❌ [ERROR] User path not found: {model_path}")

    possible_paths = [
        PROJECT_ROOT / "pretrain_models" / model_name,
        PROJECT_ROOT.parent / "models" / model_name,
    ]
    for path in possible_paths:
        if path.exists():
            print(f"✅ [SUCCESS] Found model at alternative: {path}")
            return str(path)

    raise FileNotFoundError(
        f"❌ [ERROR] Local model not found: {model_name}\n"
        f"Checked:\n"
        f"  - {PROJECT_ROOT / 'models' / model_name}\n"
        f"  - {Path(model_dir) / model_name if model_dir else '(no model_dir)'}\n"
        f"  - {PROJECT_ROOT / 'pretrain_models' / model_name}\n"
        f"  - {PROJECT_ROOT.parent / 'models' / model_name}\n"
    )


# =========================
# Tokenizer Builder
# =========================
def build_tokenizer(model_path: str):
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, use_fast=True)

    special_tokens = {
        "additional_special_tokens": [
            "[METHOD]", "[RESULT]", "[CONTRIBUTION]", "[GENERAL]",
            "[ASP]", "[/ASP]", "[OPN]", "[/OPN]", "[CAT]", "[/CAT]",
            "[POL]", "[/POL]",
            "[POLMASK]",
        ]
    }
    added = tokenizer.add_special_tokens(special_tokens)
    print(f"🔤 [INFO] Added special tokens: {added}")

    # ✅ 立刻检查：这些 token 不能是 UNK
    unk = tokenizer.unk_token_id
    for t in special_tokens["additional_special_tokens"]:
        tid = tokenizer.convert_tokens_to_ids(t)
        assert tid != unk, f"{t} is UNK! tokenizer didn't add it properly."

    method_id = tokenizer.convert_tokens_to_ids("[METHOD]")
    result_id = tokenizer.convert_tokens_to_ids("[RESULT]")
    contrib_id = tokenizer.convert_tokens_to_ids("[CONTRIBUTION]")
    general_id = tokenizer.convert_tokens_to_ids("[GENERAL]")
    print(f"🏷️ [INFO] Special token IDs - METHOD: {method_id}, RESULT: {result_id}, CONTRIBUTION: {contrib_id}, GENERAL: {general_id}")

    return tokenizer, method_id, result_id, contrib_id, general_id


# =========================
# Config
# =========================
class QuadAspectEnhancedBertConfig(PretrainedConfig):
    def __init__(
        self,
        num_labels: int = 3,
        label_smoothing: float = 0.0,
        multitask: bool = False,
        use_refined_logits: bool = False,
        backbone_model: str = "roberta-base",
        model_dir: str = "models",
        hidden_size: Optional[int] = None,
        hidden_dropout_prob: float = 0.1,
        num_categories: int = 6,
        method_token_id: Optional[int] = None,
        result_token_id: Optional[int] = None,
        contribution_token_id: Optional[int] = None,
        general_token_id: Optional[int] = None,
        ablation_mode: str = "full",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.model_dir = model_dir
        self.backbone_model = backbone_model
        model_path = resolve_local_model_path(model_dir, backbone_model)
        backbone_config = AutoConfig.from_pretrained(model_path, local_files_only=True)

        self.hidden_size = hidden_size or backbone_config.hidden_size
        self.hidden_dropout_prob = hidden_dropout_prob

        self.num_labels = num_labels
        self.multitask = multitask
        self.use_refined_logits = use_refined_logits
        self.label_smoothing = label_smoothing
        self.num_categories = num_categories

        self.method_token_id = method_token_id
        self.result_token_id = result_token_id
        self.contribution_token_id = contribution_token_id
        self.general_token_id = general_token_id
        self.ablation_mode = ablation_mode

# =========================
# Modules
# =========================
class NonLinearFusion(nn.Module):
    def __init__(self, hidden_size: int, dropout_prob: float):
        super().__init__()
        self.gate_text = nn.Linear(hidden_size, hidden_size)
        self.gate_quad = nn.Linear(hidden_size, hidden_size)
        self.gate_attn = nn.Linear(hidden_size, hidden_size)

        self.transform_text = nn.Linear(hidden_size, hidden_size)
        self.transform_quad = nn.Linear(hidden_size, hidden_size)
        self.transform_attn = nn.Linear(hidden_size, hidden_size)

        self.layer_norm = nn.LayerNorm(hidden_size * 3)

        self.fusion_layer = nn.Sequential(
            nn.Linear(hidden_size * 3, hidden_size * 4),
            nn.LayerNorm(hidden_size * 4),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_size * 4, hidden_size * 3),
            nn.LayerNorm(hidden_size * 3),
        )

        self.final_proj = nn.Sequential(
            nn.Linear(hidden_size * 4, hidden_size * 3),
            nn.LayerNorm(hidden_size * 3),
            nn.Dropout(dropout_prob),
        )

        self.extra_gate_text = nn.Linear(hidden_size, hidden_size)
        self.extra_gate_quad = nn.Linear(hidden_size, hidden_size)
        self.extra_gate_attn = nn.Linear(hidden_size, hidden_size)

    def _ensure_2d(self, tensor, mask=None, name="tensor"):
        if tensor is None:
            raise ValueError(f"{name} is None")
        if tensor.dim() == 2:
            return tensor
        if tensor.dim() == 3:
            if mask is not None:
                return masked_mean_pool(tensor, mask)
            return tensor.mean(dim=1)
        raise ValueError(f"{name} must be 2D or 3D, got {tensor.dim()}D")

    def forward(self, text_features, quad_features, shared_features, text_mask=None, quad_mask=None, attn_mask=None):
        text_features = self._ensure_2d(text_features, text_mask, "text_features")
        quad_features = self._ensure_2d(quad_features, quad_mask, "quad_features")
        attn_features = self._ensure_2d(shared_features, attn_mask, "attn_features")

        text_gate = torch.sigmoid(self.gate_text(text_features))
        quad_gate = torch.sigmoid(self.gate_quad(quad_features))
        attn_gate = torch.sigmoid(self.gate_attn(attn_features))

        text_transformed = self.transform_text(text_features)
        quad_transformed = self.transform_quad(quad_features)
        attn_transformed = self.transform_attn(attn_features)

        text_gated = text_gate * text_transformed
        quad_gated = quad_gate * quad_transformed
        attn_gated = attn_gate * attn_transformed

        combined = torch.cat([text_gated, quad_gated, attn_gated], dim=-1)
        normalized = self.layer_norm(combined)
        fused = self.fusion_layer(normalized)

        w_text = torch.sigmoid(self.extra_gate_text(attn_features))
        w_quad = torch.sigmoid(self.extra_gate_quad(attn_features))
        w_attn = torch.sigmoid(self.extra_gate_attn(attn_features))
        extra = (w_text * text_features) + (w_quad * quad_features) + (w_attn * attn_features)

        final_fusion = self.final_proj(torch.cat([fused, extra], dim=-1)) + fused
        return final_fusion


# =========================
# Model
# =========================
class QuadAspectEnhancedBertModel(PreTrainedModel):
    config_class = QuadAspectEnhancedBertConfig

    def __init__(self, config: QuadAspectEnhancedBertConfig):
        super().__init__(config)

        model_path = resolve_local_model_path(config.model_dir, config.backbone_model)
        print(f"🤖 [INFO] Loading model from: {model_path}")

        self.bert = AutoModel.from_pretrained(model_path, local_files_only=True)
        self.quad_bert = AutoModel.from_pretrained(model_path, local_files_only=True)

        hidden_size = config.hidden_size
        self.category_embedding = nn.Embedding(config.num_categories, hidden_size)

        self.text_transform = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(config.hidden_dropout_prob),
        )
        self.quad_transform = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(config.hidden_dropout_prob),
        )

        self.fusion = NonLinearFusion(hidden_size, config.hidden_dropout_prob)
        self.classifier = nn.Linear(hidden_size * 3, config.num_labels)
        self.init_weights()

    def resize_token_embeddings_after_init(self, vocab_size: int):
        print(f"🔠 [INFO] Resizing token embeddings to: {vocab_size}")
        self.bert.resize_token_embeddings(vocab_size)
        self.quad_bert.resize_token_embeddings(vocab_size)

    @staticmethod
    def _pool(outputs):
        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            return outputs.pooler_output
        return outputs.last_hidden_state[:, 0]

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            quad_input_ids=None,
            quad_attention_mask=None,
            category_ids=None,
    ):
        # text branch
        text_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        text_pooled = self._pool(text_outputs)
        text_pooled = self.text_transform(text_pooled)

        # quad branch
        quad_outputs = self.quad_bert(input_ids=quad_input_ids, attention_mask=quad_attention_mask)
        quad_hidden = quad_outputs.last_hidden_state

        if category_ids is not None:
            category_embeds = self.category_embedding(category_ids)
            category_embeds = category_embeds.unsqueeze(1).expand(-1, quad_hidden.size(1), -1)
            quad_hidden = quad_hidden + category_embeds

        quad_hidden = self.quad_transform(quad_hidden)
        quad_pooled = masked_mean_pool(quad_hidden, quad_attention_mask)

        # ablation control
        if self.config.ablation_mode == "quad_only":
            fusion_text = torch.zeros_like(text_pooled)
            fusion_quad = quad_pooled
            shared_features = quad_pooled

        elif self.config.ablation_mode == "text_only":
            fusion_text = text_pooled
            fusion_quad = torch.zeros_like(quad_pooled)
            shared_features = text_pooled

        else:  # full
            fusion_text = text_pooled
            fusion_quad = quad_pooled
            shared_features = (text_pooled + quad_pooled) / 2.0

        fused = self.fusion(
            text_features=fusion_text,
            quad_features=fusion_quad,
            shared_features=shared_features
        )

        logits = self.classifier(fused)

        return {
            "logits": logits,
            "embeddings": fused,
            "text_pooled": text_pooled,
        }


# =========================
# Trainer
# =========================
class AspectAwareTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs["logits"]
        loss = F.cross_entropy(logits, labels, label_smoothing=model.config.label_smoothing)
        return (loss, outputs) if return_outputs else loss


def compute_metrics(eval_pred):
    preds, labels = eval_pred

    # ⚠️ Trainer 可能返回 tuple（logits, extra_outputs...）
    if isinstance(preds, (tuple, list)):
        preds = preds[0]   # 只取 logits

    pred_labels = np.argmax(preds, axis=-1)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        pred_labels,
        average="macro",
        zero_division=0
    )
    acc = accuracy_score(labels, pred_labels)

    return {
        "Precision_Macro": precision,
        "Recall_Macro": recall,
        "F1_Macro": f1,
        "Accuracy": acc,
    }



# =========================
# Data Processor
# =========================
class QuadAspectDataProcessor:

    def __init__(
        self,
        tokenizer: Any,
        max_length: int = 512,
        category2id: Optional[Dict[str, int]] = None
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.category2id = category2id or {
            "METHODOLOGY": 0,
            "PERFORMANCE": 1,
            "INNOVATION": 2,
            "APPLICABILITY": 3,
            "LIMITATION": 4,
            "COMPARISON": 5
        }

    def _get_view_token(self, category: str) -> str:
        if category == "METHODOLOGY":
            return "[METHOD]"
        elif category == "PERFORMANCE":
            return "[RESULT]"
        elif category == "INNOVATION":
            return "[CONTRIBUTION]"
        return "[GENERAL]"

    def _build_quad_text_from_quads(self, quads: List[Any], sample_text: str = "") -> Tuple[str, int]:

        if quads is None:
            raise ValueError(f"[DATA ERROR] quads is None | text={sample_text[:120]}")
        if not isinstance(quads, list):
            raise ValueError(f"[DATA ERROR] quads must be list, got {type(quads)} | text={sample_text[:120]}")

        quad_parts: List[str] = []
        feature_categories: List[str] = []

        for qi, q in enumerate(quads):
            if not isinstance(q, (list, tuple)):
                raise ValueError(
                    f"[DATA ERROR] quad[{qi}] must be list/tuple, got {type(q)} | quad={q} | text={sample_text[:120]}"
                )

            if len(q) == 4:
                aspect, opinion, category, polarity = q
            elif len(q) == 3:
                aspect, opinion, category = q
                polarity = None
            else:
                raise ValueError(
                    f"[DATA ERROR] quad[{qi}] length must be 3 or 4, got len={len(q)} | quad={q} | text={sample_text[:120]}"
                )

            aspect = str(aspect).strip() if aspect is not None else ""
            opinion = str(opinion).strip() if opinion is not None else ""
            category = str(category).strip().upper() if category is not None else ""

            if not aspect:
                raise ValueError(
                    f"[DATA ERROR] Empty aspect in quad[{qi}] | quad={q} | text={sample_text[:120]}"
                )
            if not opinion:
                raise ValueError(
                    f"[DATA ERROR] Empty opinion in quad[{qi}] | quad={q} | text={sample_text[:120]}"
                )
            if category not in self.category2id:
                raise ValueError(
                    f"[DATA ERROR] Unknown category='{category}' in quad[{qi}] | quad={q} | text={sample_text[:120]}"
                )

            view_token = self._get_view_token(category)
            polarity_token = "[POLMASK]"

            quad_parts.append(
                f"{view_token} "
                f"[ASP] {aspect} [/ASP] "
                f"[OPN] {opinion} [/OPN] "
                f"[CAT] {category} [/CAT] "
                f"[POL] {polarity_token} [/POL]"
            )
            feature_categories.append(category)

        if len(quad_parts) == 0:
            raise ValueError(f"[DATA ERROR] No valid quads after parsing (empty quad_parts) | text={sample_text[:120]}")

        quad_text_local = " ; ".join(quad_parts)

        # 主导类别：出现最多的 category
        major_cat = Counter(feature_categories).most_common(1)[0][0]
        category_id_local = self.category2id[major_cat]

        return quad_text_local, category_id_local

    def process_features(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        texts = [f.get("text", "") for f in features]
        labels = [int(f.get("label", 0)) for f in features]

        text_encoding = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        quad_texts = []
        category_ids = []

        for feature in features:
            label = int(feature.get("label", 0))
            text = feature.get("text", "")
            quads = feature.get("quads", [])
            if label == 0:
                if not quads:
                    raise ValueError(f"[DATA] Neutral sample missing quads: {text[:120]}")
                qt, cid = self._build_quad_text_from_quads(quads, sample_text=text)
                if not qt:
                    raise ValueError(f"[DATA] Neutral quad_text empty after build. quads={quads} text={text[:120]}")
                quad_texts.append(qt)
                category_ids.append(cid)
                continue

            if not quads:
                raise ValueError(f"[DATA] Missing quads for text: {text[:120]}")

            qt, cid = self._build_quad_text_from_quads(quads, sample_text=text)
            if not qt:
                raise ValueError(f"[DATA] Failed to build quad_text from quads: {quads} | text: {text[:120]}")

            quad_texts.append(qt)
            category_ids.append(cid)

        quad_encoding = self.tokenizer(
            quad_texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        return {
            "input_ids": text_encoding["input_ids"],
            "attention_mask": text_encoding["attention_mask"],
            "quad_input_ids": quad_encoding["input_ids"],
            "quad_attention_mask": quad_encoding["attention_mask"],
            "category_ids": torch.tensor(category_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


# =========================
# Dataset
# =========================
class AspectAwareDataset(torch.utils.data.Dataset):
    def __init__(self, samples, processor: QuadAspectDataProcessor, do_shuffle: bool = False):
        label_map = {"neutral": 0, "positive": 1, "negative": 2}

        raw = []
        for item in samples:
            sent = item.get("overall_sentiment", "").lower().strip()
            if sent not in label_map:
                continue
            raw.append({
                "text": item.get("text", ""),
                "label": label_map[sent],
                "quads": item.get("quads", item.get("sentiment_quadruples", [])) or [],
            })

        if do_shuffle:
            random.shuffle(raw)

        processed = processor.process_features(raw)

        # ✅ 改成 list，每个元素是一个 dict（更省内存）
        self.items = []
        n = processed["input_ids"].size(0)
        for i in range(n):
            self.items.append({k: v[i] for k, v in processed.items()})

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


def build_dataset_from_samples(samples, tokenizer, args, is_train: bool):
    processor = QuadAspectDataProcessor(
        tokenizer,
        max_length=args.max_length,
    )
    # do_shuffle 必须 False，保证顺序可控（OOF 要对齐 index）
    return AspectAwareDataset(samples, processor=processor, do_shuffle=False)



# =========================
# Data Loader (NEW)
# =========================
def load_merged_dataset(merged_json_path: str):
    """
    merged_json: list[dict] with keys:
      - text
      - overall_sentiment: positive|negative|neutral
      - sentiment_quadruples: [[aspect, opinion, category, polarity], ...]
    Return:
      samples: list[dict] each has {text, overall_sentiment, quads}
    """
    with open(merged_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    samples = []
    for item in data:
        sent = (item.get("overall_sentiment") or "").strip().lower()
        text = (item.get("text") or "").strip()
        if not text:
            continue
        if sent not in ("positive", "negative", "neutral"):
            continue

        samples.append({
            "text": text,
            "overall_sentiment": sent,
            "quads": item.get("sentiment_quadruples", []) or [],
        })

    return samples




# =========================
# Train / Inference Utils
# =========================
def train_asqp_model(args, train_data, eval_data, tokenizer, method_id, result_id, contrib_id, general_id, run_tag: str = ""):
    config = QuadAspectEnhancedBertConfig(
        num_labels=3,
        label_smoothing=0.0,
        multitask=False,
        use_refined_logits=False,
        backbone_model=args.model_name,
        model_dir=args.model_dir,
        num_categories=6,
        method_token_id=method_id,
        result_token_id=result_id,
        contribution_token_id=contrib_id,
        general_token_id=general_id,
        ablation_mode=args.ablation_mode,
    )

    model = QuadAspectEnhancedBertModel(config)
    model.resize_token_embeddings_after_init(len(tokenizer))
    model.to(args.device)
    tag = run_tag or "single"
    results_dir = PROJECT_ROOT / "results" / args.model_name / args.ablation_mode / tag
    logs_dir = PROJECT_ROOT / "logs" / args.model_name / args.ablation_mode / tag
    results_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    do_eval = eval_data is not None

    training_args = TrainingArguments(
        seed=args.seed,
        report_to="none",
        output_dir=str(results_dir),
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        lr_scheduler_type="cosine",
        gradient_accumulation_steps=args.accumulation_steps,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        logging_strategy="steps",
        logging_steps=50,

        # 🔥 关键修改
        evaluation_strategy="no" ,
        save_strategy="no",
        remove_unused_columns=False,
        bf16=bool(args.bf16 and torch.cuda.is_available()),
        dataloader_num_workers=0,
    )

    trainer = AspectAwareTrainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        data_collator=default_data_collator,
        train_dataset=train_data,
        eval_dataset=eval_data if do_eval else None,
        compute_metrics= None,
    )

    trainer.train()
    clean_memory()
    return model


@torch.no_grad()
def predict_probs(model, dataset, device="cuda", batch_size=32):
    model.eval()
    model.to(device)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_probs = []
    all_labels = []

    for batch in loader:
        inputs = {k: v.to(device) for k, v in batch.items() if k != "labels"}
        outputs = model(**inputs)
        logits = outputs["logits"]

        if getattr(model.config, "use_refined_logits", False):
            probs = torch.exp(logits)
        else:
            probs = torch.softmax(logits, dim=-1)

        all_probs.append(probs.cpu().numpy())
        all_labels.append(batch["labels"].cpu().numpy())
    clean_memory()
    return np.concatenate(all_probs, axis=0), np.concatenate(all_labels, axis=0)

def tune_ensemble_weights(val_probs_list, val_labels):
    """
    目标：优化模型权重，以获得最佳的F1分数。
    通过贝叶斯优化方法 `gp_minimize` 找到最优的集成权重。
    """

    def objective(weights):
        weights = np.array(weights, dtype=np.float32)
        s = weights.sum()
        if s <= 0:
            return 100.0  # penalty，避免非法解

        # 权重归一化
        weights = weights / s

        final_probs = np.zeros_like(val_probs_list[0])
        for p, w in zip(val_probs_list, weights):
            final_probs += w * p

        preds = np.argmax(final_probs, axis=1)
        return -f1_score(val_labels, preds, average="macro")  # 以F1分数作为优化目标

    space = [Real(0.0, 1.0, name=f"w{i}") for i in range(len(val_probs_list))]

    # 贝叶斯优化
    res = gp_minimize(objective, space, n_calls=50, random_state=42)

    best_weights = np.array(res.x, dtype=np.float32)
    best_weights = best_weights / (best_weights.sum() + 1e-12)

    return best_weights.tolist(), -res.fun  # 返回权重和F1分数

def weighted_ensemble(probs_list, weights):
    weights = np.array(weights, dtype=np.float32)
    s = float(weights.sum())
    if s <= 0:
        raise ValueError(f"Invalid weights sum={s}, weights={weights}")
    weights = weights / s
    final_probs = np.zeros_like(probs_list[0])
    for p, w in zip(probs_list, weights):
        final_probs += w * p
    return np.argmax(final_probs, axis=1)

def safe_copy_args(args, **overrides):
    d = vars(args).copy()
    d.update(overrides)
    return argparse.Namespace(**d)

ENSEMBLE_MODELS = [
    {"name": "bert-base-uncased"},
    {"name": "roberta-base"},
    {"name": "electra-base-discriminator"},
]

# =========================
# Main
# =========================
# =========================
# Main
# =========================
def main(args):
    print("=" * 60)
    print("🚀 QUAD ASPECT ENHANCED BERT TRAINING (merged JSON, offline)")
    print("=" * 60)

    merged_path = (PROJECT_ROOT / args.merged_json).resolve() if not Path(args.merged_json).is_absolute() else Path(args.merged_json)
    if not merged_path.exists():
        print(f"❌ [ERROR] merged_json not found: {merged_path}")
        return

    print(f"✅ [INFO] Merged dataset: {merged_path}")

    # -------------------------
    # 0) Load
    # -------------------------
    samples = load_merged_dataset(str(merged_path))
    dist = Counter(s["overall_sentiment"] for s in samples)
    print(f"📊 [INFO] Loaded total={len(samples)} | neutral={dist.get('neutral', 0)} pos={dist.get('positive', 0)} neg={dist.get('negative', 0)}")

    # 2) Build OUTER 10-fold splits
    # -------------------------
    label_map = {"neutral": 0, "positive": 1, "negative": 2}
    y = np.array([label_map[s["overall_sentiment"]] for s in samples], dtype=np.int64)
    n_total = len(samples)

    print("\n📈 [INFO] Dataset distribution (ALL):")
    print(f"   ALL: Neutral={dist.get('neutral', 0)} Positive={dist.get('positive', 0)} Negative={dist.get('negative', 0)}")

    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=args.seed)
    folds = list(skf.split(np.arange(n_total), y))

    print("\n" + "=" * 60)
    print("📌 OUTER 10-FOLD + INNER HOLD-OUT (tune on inner-val, RETRAIN on full outer-train, eval on outer-test)")
    print("=" * 60)

    outer_fold_accs = []
    outer_fold_f1s = []
    outer_fold_weights = []  # 这里初始化 outer_fold_weights

    for fold_id, (tr_idx, te_idx) in enumerate(folds, start=1):
        print(f"\n==================== OUTER Fold {fold_id}/10 ====================")

        # ----- inner split inside outer-train -----
        y_tr = y[tr_idx]
        sss = StratifiedShuffleSplit(
            n_splits=1,
            test_size=0.1,  # 可改 0.2 更稳
            random_state=args.seed + fold_id
        )
        inner_tr_rel, inner_val_rel = next(sss.split(np.zeros(len(tr_idx)), y_tr))
        inner_tr_idx = tr_idx[inner_tr_rel]
        inner_val_idx = tr_idx[inner_val_rel]

        train_inner_samples = [samples[i] for i in inner_tr_idx]
        val_inner_samples = [samples[i] for i in inner_val_idx]
        test_outer_samples = [samples[i] for i in te_idx]

        val_labels = y[inner_val_idx]
        test_labels = y[te_idx]

        print(f"[Split] inner-train={len(inner_tr_idx)} | inner-val={len(inner_val_idx)} | outer-test={len(te_idx)}")

        # ========== (1) 用 inner-train 训练 -> 在 inner-val 上出 probs，用来调权重 ==========
        val_probs_list = []
        for model_cfg in ENSEMBLE_MODELS:
            backbone = model_cfg["name"]
            print(f"\n🔥 [Outer {fold_id}] (Tune) Train backbone on inner-train: {backbone}")

            fold_args = safe_copy_args(args, model_name=backbone)

            model_path = resolve_local_model_path(fold_args.model_dir, fold_args.model_name)
            tokenizer, method_id, result_id, contrib_id, general_id = build_tokenizer(model_path)

            train_inner_dataset = build_dataset_from_samples(train_inner_samples, tokenizer, fold_args, is_train=True)
            val_inner_dataset = build_dataset_from_samples(val_inner_samples, tokenizer, fold_args, is_train=False)

            model = train_asqp_model(
                fold_args,
                train_data=train_inner_dataset,
                eval_data=val_inner_dataset,
                tokenizer=tokenizer,
                method_id=method_id,
                result_id=result_id,
                contrib_id=contrib_id,
                general_id=general_id,
                run_tag=f"outer_{fold_id}_tune_inner_{backbone}",
            )

            # inner-val probs (ONLY for tuning weights)
            val_probs, val_y_check = predict_probs(
                model,
                val_inner_dataset,
                device=fold_args.device,
                batch_size=fold_args.batch_size
            )
            if not np.array_equal(val_y_check, val_labels):
                raise ValueError(f"[ALIGN ERROR] outer={fold_id} backbone={backbone} inner-val labels mismatch")

            val_probs_list.append(val_probs)

            # cleanup
            del model, train_inner_dataset, val_inner_dataset, tokenizer
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # ----- (2) 调权重：只用 inner-val -----
        w_fold, best_val_f1 = tune_ensemble_weights(val_probs_list, val_labels)
        w_norm = np.array(w_fold, dtype=np.float32)
        w_norm = (w_norm / (w_norm.sum() + 1e-12)).tolist()
        print(f"\n🎯 [Outer {fold_id}] tuned weights={w_norm} | bestF1(inner-val)={best_val_f1:.4f}")

        # ========== (3) 固定权重后：用 full outer-train 重训 backbone，再在 outer-test 上测 ==========
        full_train_samples = [samples[i] for i in tr_idx]
        test_probs_list = []

        for model_cfg in ENSEMBLE_MODELS:
            backbone = model_cfg["name"]
            print(f"\n🔥 [Outer {fold_id}] (Final) Retrain backbone on FULL outer-train: {backbone}")

            fold_args = safe_copy_args(args, model_name=backbone)

            model_path = resolve_local_model_path(fold_args.model_dir, fold_args.model_name)
            tokenizer, method_id, result_id, contrib_id, general_id = build_tokenizer(model_path)

            full_train_dataset = build_dataset_from_samples(full_train_samples, tokenizer, fold_args, is_train=True)
            test_outer_dataset = build_dataset_from_samples(test_outer_samples, tokenizer, fold_args, is_train=False)

            model = train_asqp_model(
                fold_args,
                train_data=full_train_dataset,
                eval_data=None,
                tokenizer=tokenizer,
                method_id=method_id,
                result_id=result_id,
                contrib_id=contrib_id,
                general_id=general_id,
                run_tag=f"outer_{fold_id}_final_fulltrain_{backbone}",
            )

            te_probs, te_y_check = predict_probs(
                model,
                test_outer_dataset,
                device=fold_args.device,
                batch_size=fold_args.batch_size
            )
            if not np.array_equal(te_y_check, test_labels):
                raise ValueError(f"[ALIGN ERROR] outer={fold_id} backbone={backbone} outer-test labels mismatch")

            test_probs_list.append(te_probs)

            # cleanup
            del model, full_train_dataset, test_outer_dataset, tokenizer
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # ----- (4) outer-test ONLY evaluation -----
        test_preds = weighted_ensemble(test_probs_list, w_norm)
        fold_acc = accuracy_score(test_labels, test_preds)
        fold_f1 = f1_score(test_labels, test_preds, average="macro")

        outer_fold_weights.append(w_norm)  # 添加到列表中
        outer_fold_accs.append(fold_acc)
        outer_fold_f1s.append(fold_f1)

        print(f"\n✅ [Outer Fold {fold_id:02d}] "
              f"Acc(outer-test)={fold_acc:.4f} | Macro-F1(outer-test)={fold_f1:.4f}")
    clean_memory()

    # 只计算准确度和F1分数的平均值
    mean_acc = np.mean(outer_fold_accs)
    mean_f1 = np.mean(outer_fold_f1s)

    print("\n" + "=" * 60)
    print("✅ OUTER 10-FOLD SUMMARY (mean over outer-test folds) [STRICT NO-LEAKAGE]")
    print("=" * 60)
    print(f"Accuracy: {mean_acc:.4f}")
    print(f"Macro-F1: {mean_f1:.4f}")

    print("\n📌 Per-outer-fold tuned weights (normalized; tuned on inner-val only):")
    for i, w in enumerate(outer_fold_weights, start=1):
        print(f"  Outer Fold {i:02d}: {w}")



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
    parser.add_argument("--ablation_mode",type=str,default="full",choices=["full", "quad_only", "text_only"],help="Ablation setting for model inputs.")
    args = parser.parse_args()

    print("⚙️ [INFO] Training configuration:")
    print(f"   Merged dataset: {args.merged_json}")
    print(f"   Model dir: {args.model_dir}")
    print(f"   Epochs: {args.epochs}")
    print(f"   LR: {args.learning_rate}")
    print(f"   Batch size: {args.batch_size}")
    print(f"   Device: {args.device}")
    print(f"   Ablation mode: {args.ablation_mode}")
    seed_everything(args.seed)
    main(args)
