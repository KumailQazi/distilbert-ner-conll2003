#!/usr/bin/env python3
"""Fine-tune DistilBERT for CoNLL-2003 NER with reproducible token alignment."""
from __future__ import annotations

import argparse
import json
import os
import platform
import resource
import sys
import time
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.json"))
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
        f.write("\n")


def cpu_model_name() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def total_memory_gib() -> float | None:
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("MemTotal:"):
                return round(int(line.split()[1]) / (1024**2), 2)
    except (OSError, ValueError, IndexError):
        pass
    return None


def peak_rss_mib() -> float | None:
    try:
        value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # Linux reports KiB; macOS reports bytes.
        divisor = 1024**2 if sys.platform == "darwin" else 1024
        return round(value / divisor, 1)
    except (AttributeError, OSError):
        return None


def package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def main() -> None:
    args = parse_args()
    config_path = args.config.expanduser().resolve()
    config = read_json(config_path)
    repo_root = Path(__file__).resolve().parent
    output_dir = (args.output_dir or (repo_root / config["output_dir"])).expanduser().resolve()
    cache_dir = (repo_root / config["cache_dir"]).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Set cache path before loading Hub/Datasets components.
    os.environ.setdefault("HF_HOME", str(cache_dir))
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    import numpy as np
    import torch
    from datasets import load_dataset
    from huggingface_hub import hf_hub_download
    from seqeval.metrics import accuracy_score, f1_score, precision_score, recall_score
    from transformers import (
        AutoModelForTokenClassification,
        AutoTokenizer,
        DataCollatorForTokenClassification,
        Trainer,
        TrainingArguments,
        set_seed,
    )

    seed = int(config["seed"])
    set_seed(seed)
    torch.set_num_threads(int(config.get("torch_num_threads", os.cpu_count() or 1)))
    force_cpu = bool(config.get("force_cpu", True))
    use_cpu = force_cpu or not torch.cuda.is_available()
    device_name = "CPU" if use_cpu else torch.cuda.get_device_name(0)

    # Download the raw JSONL files at an immutable Hub commit and load them as
    # plain JSON. This avoids executing the dataset repository's Python loader.
    dataset_repo = config["dataset_repo_id"]
    dataset_revision = config["dataset_revision"]
    split_files = {
        "train": "dataset/train.json",
        "validation": "dataset/valid.json",
        "test": "dataset/test.json",
    }
    local_files = {
        split: hf_hub_download(
            repo_id=dataset_repo,
            repo_type="dataset",
            filename=filename,
            revision=dataset_revision,
            cache_dir=str(cache_dir),
        )
        for split, filename in split_files.items()
    }
    label_path = hf_hub_download(
        repo_id=dataset_repo,
        repo_type="dataset",
        filename="dataset/label.json",
        revision=dataset_revision,
        cache_dir=str(cache_dir),
    )
    raw_label_map = read_json(Path(label_path))
    id2label = {int(label_id): label for label, label_id in raw_label_map.items()}
    label_names = [id2label[i] for i in range(len(id2label))]
    if label_names != config["label_names"]:
        raise ValueError(f"Pinned dataset labels do not match config: {label_names}")
    label2id = {label: idx for idx, label in id2label.items()}

    data_files = {split: path for split, path in local_files.items()}
    datasets = load_dataset(
        "json",
        data_files=data_files,
        cache_dir=str(cache_dir),
    )
    for split in ("train", "validation", "test"):
        if not datasets[split]:
            raise ValueError(f"The {split} split is empty")

    model_name = config["model_name_or_path"]
    model_revision = config["model_revision"]
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        revision=model_revision,
        use_fast=True,
        cache_dir=str(cache_dir),
    )
    if not tokenizer.is_fast:
        raise RuntimeError("A fast tokenizer is required for word_ids-based alignment")

    max_length = int(config["max_sequence_length"])

    def tokenize_and_align_labels(batch: dict[str, list[Any]]) -> dict[str, Any]:
        tokenized = tokenizer(
            batch["tokens"],
            is_split_into_words=True,
            truncation=True,
            max_length=max_length,
        )
        aligned_labels = []
        for batch_index, word_labels in enumerate(batch["tags"]):
            word_ids = tokenized.word_ids(batch_index=batch_index)
            previous_word_id = None
            label_ids = []
            for word_id in word_ids:
                if word_id is None or word_id == previous_word_id:
                    # Ignore special tokens and non-initial wordpieces in loss/metrics.
                    label_ids.append(-100)
                else:
                    label_ids.append(int(word_labels[word_id]))
                previous_word_id = word_id
            aligned_labels.append(label_ids)
        tokenized["labels"] = aligned_labels
        return tokenized

    tokenized_datasets = datasets.map(
        tokenize_and_align_labels,
        batched=True,
        remove_columns=datasets["train"].column_names,
        desc="Tokenize and align NER labels",
    )

    def compute_metrics(eval_prediction: Any) -> dict[str, float]:
        logits, label_ids = eval_prediction
        if isinstance(logits, tuple):
            logits = logits[0]
        predicted_ids = np.argmax(logits, axis=-1)
        true_predictions = []
        true_labels = []
        for prediction_row, label_row in zip(predicted_ids, label_ids):
            row_predictions = []
            row_labels = []
            for prediction_id, label_id in zip(prediction_row, label_row):
                if int(label_id) == -100:
                    continue
                row_predictions.append(id2label[int(prediction_id)])
                row_labels.append(id2label[int(label_id)])
            true_predictions.append(row_predictions)
            true_labels.append(row_labels)
        return {
            "precision": float(precision_score(true_labels, true_predictions)),
            "recall": float(recall_score(true_labels, true_predictions)),
            "f1": float(f1_score(true_labels, true_predictions)),
            "token_accuracy": float(accuracy_score(true_labels, true_predictions)),
        }

    model = AutoModelForTokenClassification.from_pretrained(
        model_name,
        revision=model_revision,
        num_labels=len(label_names),
        id2label=id2label,
        label2id=label2id,
        cache_dir=str(cache_dir),
    )
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        learning_rate=float(config["learning_rate"]),
        per_device_train_batch_size=int(config["per_device_train_batch_size"]),
        per_device_eval_batch_size=int(config["per_device_eval_batch_size"]),
        num_train_epochs=float(config["num_train_epochs"]),
        weight_decay=float(config["weight_decay"]),
        warmup_ratio=float(config["warmup_ratio"]),
        logging_strategy="steps",
        logging_steps=int(config["logging_steps"]),
        report_to="none",
        seed=seed,
        data_seed=seed,
        use_cpu=use_cpu,
        fp16=False,
        dataloader_num_workers=int(config.get("dataloader_num_workers", 0)),
        dataloader_pin_memory=False,
        save_safetensors=True,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    started_utc = datetime.now(timezone.utc).isoformat()
    run_started = time.perf_counter()
    train_started = run_started
    train_result = trainer.train()
    training_seconds = time.perf_counter() - train_started
    validation_metrics = trainer.evaluate(
        eval_dataset=tokenized_datasets["validation"],
        metric_key_prefix="validation",
    )
    test_metrics = trainer.evaluate(
        eval_dataset=tokenized_datasets["test"],
        metric_key_prefix="test",
    )
    trainer.save_model(str(output_dir / "best_model"))
    tokenizer.save_pretrained(str(output_dir / "best_model"))
    total_seconds = time.perf_counter() - run_started

    metrics = {
        "started_utc": started_utc,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "training_seconds": round(training_seconds, 3),
        "total_run_seconds": round(total_seconds, 3),
        "train_samples": len(datasets["train"]),
        "validation_samples": len(datasets["validation"]),
        "test_samples": len(datasets["test"]),
        "train_runtime_metrics": train_result.metrics,
        "validation": validation_metrics,
        "test": test_metrics,
        "hardware": {
            "platform": platform.platform(),
            "cpu_model": cpu_model_name(),
            "logical_cpu_count": os.cpu_count(),
            "system_ram_gib": total_memory_gib(),
            "device": device_name,
            "peak_process_rss_mib": peak_rss_mib(),
            "torch_cuda_available": torch.cuda.is_available(),
        },
        "software": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": package_version("transformers"),
            "datasets": package_version("datasets"),
            "seqeval": package_version("seqeval"),
            "accelerate": package_version("accelerate"),
        },
    }
    resolved_config = dict(config)
    resolved_config["actual_data_files"] = split_files
    resolved_config["resolved_labels"] = id2label
    resolved_config["run_output_dir"] = str(output_dir)
    resolved_config["actual_runtime"] = {
        "device": device_name,
        "logical_cpu_count": os.cpu_count(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": package_version("transformers"),
        "datasets": package_version("datasets"),
        "seqeval": package_version("seqeval"),
        "accelerate": package_version("accelerate"),
    }
    write_json(output_dir / "metrics.json", metrics)
    write_json(output_dir / "resolved_config.json", resolved_config)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"\nSaved run outputs to: {output_dir}")


if __name__ == "__main__":
    main()
