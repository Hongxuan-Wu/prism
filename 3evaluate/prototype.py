#!/usr/bin/env python3
"""Evaluate frozen within-dataset and cross-dataset reference prototypes."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, auc, f1_score, precision_recall_curve, roc_auc_score

MODEL_IDS = ("evo8k", "nt_2.5b", "ntv2_500m", "dnabert2", "m1", "m2", "m8", "random_genome")
BATCH_SIZES = {"evo8k": 32, "nt_2.5b": 64, "ntv2_500m": 128, "dnabert2": 256,
               "m1": 256, "m2": 256, "m8": 256, "random_genome": 256}
METRICS = ("accuracy", "f1", "auroc", "auprc", "legacy_positive_auroc", "legacy_positive_auprc")
SUFFIXES = {"train": "_catATG_train.csv", "test": "_catATG_test.csv"}
DIRECTORIES = {"train": "Train_catATG", "test": "Test_catATG"}
DNA = frozenset("ACGTRYSWKMBDHVN")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def normalized_features(features, label="features"):
    features = np.asarray(features)
    if features.ndim != 2 or not len(features) or not np.all(np.isfinite(features)):
        raise ValueError(f"{label} embeddings must be a non-empty finite matrix")
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError(f"{label} embedding has zero norm")
    return features / norms


def binary_labels(labels, count):
    labels = np.asarray(labels)
    if labels.shape != (count,) or set(np.unique(labels)) != {0, 1}:
        raise ValueError("Aligned binary labels containing both classes are required")
    return labels


def class_prototypes(reference_features, reference_labels):
    reference = normalized_features(reference_features, "reference")
    labels = binary_labels(reference_labels, len(reference))
    prototypes = []
    for label in (0, 1):
        prototype = reference[labels == label].mean(axis=0)
        norm = np.linalg.norm(prototype)
        if not norm:
            raise ValueError("Reference prototype has zero norm")
        prototypes.append(prototype / norm)
    return prototypes[0], prototypes[1]


def score_features(prototypes, test_features):
    """Construct scores without accepting test labels."""
    test = normalized_features(test_features, "test")
    if any(np.asarray(p).shape != (test.shape[1],) for p in prototypes):
        raise ValueError("Reference and test feature dimensions differ")
    negative = test @ prototypes[0]
    positive = test @ prototypes[1]
    return positive - negative, positive, negative


def metric_values(y_true, margin, positive):
    y_true = binary_labels(y_true, len(margin))
    if margin.shape != y_true.shape or positive.shape != y_true.shape:
        raise ValueError("Metric arrays have incompatible shapes")
    if not np.all(np.isfinite(margin)) or not np.all(np.isfinite(positive)):
        raise ValueError("Metric scores must be finite")

    def auprc(score):
        precision, recall, _ = precision_recall_curve(y_true, score)
        return float(auc(recall, precision))

    return {
        "accuracy": float(accuracy_score(y_true, margin > 0.0)),
        "f1": float(f1_score(y_true, margin > 0.0, zero_division=0)),
        "auroc": float(roc_auc_score(y_true, margin)),
        "auprc": auprc(margin),
        "legacy_positive_auroc": float(roc_auc_score(y_true, positive)),
        "legacy_positive_auprc": auprc(positive),
    }


def read_sequences(path):
    sequences, labels = [], []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["sequences", "labels"]:
            raise ValueError(f"Expected CSV columns sequences,labels: {path.name}")
        for row in reader:
            if None in row or any(value is None for value in row.values()):
                raise ValueError("Malformed input CSV row")
            sequence = "".join(row["sequences"].split()).upper()
            label = int(row["labels"])
            if len(sequence) != 801 or set(sequence) - DNA or label not in (0, 1):
                raise ValueError(f"Expected 801-bp IUPAC DNA and binary label: {path.name}")
            sequences.append(sequence)
            labels.append(label)
    binary_labels(labels, len(sequences))
    return sequences, labels


def discover_inputs(root, allow_subset=False):
    rosters = {}
    for split in SUFFIXES:
        rosters[split] = {
            path.name[:-len(SUFFIXES[split])]: path
            for path in sorted((root / DIRECTORIES[split]).glob("*" + SUFFIXES[split]))
        }
    if not rosters["train"] or set(rosters["train"]) != set(rosters["test"]):
        raise ValueError("Original Train_catATG and Test_catATG rosters must match")
    if not allow_subset and len(rosters["train"]) != 28:
        raise ValueError("Expected 28 species; use --allow-subset only for a labeled subset/demo")
    records = []
    totals = {"train": 0, "test": 0}
    for species in sorted(rosters["train"]):
        for split in SUFFIXES:
            path = rosters[split][species]
            sequences, labels = read_sequences(path)
            records.append({"species": species, "split": split, "path": path,
                            "rows": len(labels), "input_sha256": sha256(path)})
            totals[split] += len(labels)
    if not allow_subset and totals != {"train": 206137, "test": 51551}:
        raise ValueError(f"Original manuscript row counts differ: {totals}")
    return records


def encode(args):
    if not args.trust_checkpoint_code:
        raise ValueError("Review checkpoint Python/weights, then set --trust-checkpoint-code")
    args.batch_size = BATCH_SIZES[args.model_id] if args.batch_size is None else args.batch_size
    if args.batch_size < 1 or not args.checkpoint.is_dir():
        raise ValueError("A local checkpoint directory and positive batch size are required")
    records = discover_inputs(args.data_root, args.allow_subset)
    from frozen_encoder import extract_features, load_feature_model
    import torch
    import transformers

    torch.manual_seed(42)
    np.random.seed(42)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device(args.device)
    args.output.mkdir(parents=True, exist_ok=False)
    model, tokenizer, metadata = load_feature_model(
        args.model_id, args.checkpoint.resolve(), device, trust_checkpoint_code=True)
    checkpoint_inventory = {
        str(path.relative_to(args.checkpoint)): sha256(path)
        for path in sorted(args.checkpoint.rglob("*"))
        if path.is_file() and path.suffix in {".json", ".py", ".bin", ".pt", ".safetensors"}
    }
    files = []
    for index, record in enumerate(records):
        sequences, labels = read_sequences(record["path"])
        if sha256(record["path"]) != record["input_sha256"]:
            raise ValueError("Input changed after inventory")
        features = extract_features(args.model_id, model, tokenizer, sequences, args.batch_size, device)
        name = f"{index:04d}_{record['split']}.npz"
        np.savez(args.output / name, features=features, labels=np.asarray(labels, dtype=np.int8))
        files.append({key: value for key, value in record.items() if key != "path"})
        files[-1].update({"file": name, "sha256": sha256(args.output / name)})
        print(f"Encoded {record['species']} {record['split']}: {len(labels)} rows", flush=True)
    write_json(args.output / "manifest.json", {
        "schema": "prism-public-embeddings/v1", "status": "completed", "model_id": args.model_id,
        "split_family": "original_random_8_2", "subset_demo": args.allow_subset,
        "encoder_updates": 0, "pooling": "first_token", "transformer_max_tokens": 100,
        "evo_max_characters": 400, "seed": 42, "batch_size": args.batch_size,
        "device_type": device.type, "python": platform.python_version(), "torch": torch.__version__,
        "transformers": transformers.__version__, "numpy": np.__version__, "loader": metadata,
        "checkpoint_files_sha256": checkpoint_inventory, "files": files,
        "source_sha256": {p.name: sha256(p) for p in (Path(__file__), Path(__file__).with_name("frozen_encoder.py"))},
    })


def load_embeddings(root):
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if (manifest.get("schema") != "prism-public-embeddings/v1"
            or manifest.get("status") != "completed" or manifest.get("encoder_updates") != 0
            or manifest.get("split_family") != "original_random_8_2"
            or manifest.get("pooling") != "first_token"
            or manifest.get("transformer_max_tokens") != 100 or manifest.get("evo_max_characters") != 400
            or manifest.get("model_id") not in MODEL_IDS):
        raise ValueError("Completed original-split frozen embeddings are required")
    index = {}
    names = set()
    for record in manifest["files"]:
        name = record["file"]
        key = (record["species"], record["split"])
        if (not name or Path(name).name != name or name in names or key in index
                or record["split"] not in SUFFIXES or int(record["rows"]) < 2
                or not isinstance(record["species"], str) or not record["species"]):
            raise ValueError("Invalid or duplicate embedding inventory record")
        if (root / name).is_symlink() or sha256(root / name) != record["sha256"]:
            raise ValueError("Embedding checksum mismatch or symlink")
        names.add(name)
        index[key] = record
    train = {name for name, split in index if split == "train"}
    test = {name for name, split in index if split == "test"}
    if not train or train != test:
        raise ValueError("Embedding train/test rosters differ")
    if not manifest.get("subset_demo"):
        totals = {split: sum(r["rows"] for r in index.values() if r["split"] == split) for split in SUFFIXES}
        if len(train) != 28 or totals != {"train": 206137, "test": 51551}:
            raise ValueError("Manuscript embedding coverage differs")
    return manifest, index, sorted(train)


def read_embedding(root, record):
    with np.load(root / record["file"], allow_pickle=False) as arrays:
        features, labels = arrays["features"], arrays["labels"]
    if len(features) != record["rows"]:
        raise ValueError("Embedding row count differs from inventory")
    normalized_features(features)
    binary_labels(labels, len(features))
    return features, labels


def score(args):
    manifest, index, species = load_embeddings(args.embeddings)
    if not math.isfinite(args.diagonal_tolerance) or not 0 <= args.diagonal_tolerance <= 0.005:
        raise ValueError("Diagonal tolerance must be between 0 and 0.005")
    reference = {}
    if args.diagonal_reference:
        with args.diagonal_reference.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle, delimiter="\t"):
                if row["source_species"] != row["target_species"] or row["model_id"] != manifest["model_id"]:
                    continue
                key = (row["target_species"], row["metric_name"])
                value = float(row["value"])
                if key in reference or not math.isfinite(value) or not 0 <= value <= 1:
                    raise ValueError("Invalid or duplicate diagonal reference")
                reference[key] = value
        if set(reference) != {(name, metric) for name in species for metric in METRICS}:
            raise ValueError("Diagonal reference coverage differs")
    prototypes = {}
    for name in species:
        features, labels = read_embedding(args.embeddings, index[name, "train"])
        prototypes[name] = class_prototypes(features, labels)
    args.output.mkdir(parents=True, exist_ok=False)
    fields = ("model_id", "source_species", "target_species", "evaluation_scope", "metric_name",
              "value", "source_train_n", "target_test_n")
    summary, diagonal_errors = [], []
    with (args.output / "metrics.tsv").open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for target in species:
            features, labels = read_embedding(args.embeddings, index[target, "test"])
            target_values = {metric: [] for metric in ("auroc", "auprc")}
            for source in species if args.cross_domain else [target]:
                margin, positive, _ = score_features(prototypes[source], features)
                values = metric_values(labels, margin, positive)
                for metric, value in values.items():
                    writer.writerow({"model_id": manifest["model_id"], "source_species": source,
                                     "target_species": target, "evaluation_scope": "diagonal" if source == target else "off_diagonal",
                                     "metric_name": metric, "value": f"{value:.12g}",
                                     "source_train_n": index[source, "train"]["rows"],
                                     "target_test_n": len(labels)})
                    if source == target and reference:
                        diagonal_errors.append(abs(value - reference[target, metric]))
                if source != target:
                    for metric in target_values:
                        target_values[metric].append(values[metric])
            if args.cross_domain and len(species) > 1:
                summary.append({"target_species": target, "foreign_sources": len(species) - 1,
                                **{f"mean_{metric}": float(np.mean(values)) for metric, values in target_values.items()}})
    if diagonal_errors and max(diagonal_errors) > args.diagonal_tolerance:
        raise ValueError("Diagonal replay failed; metrics.tsv is partial/unaccepted (no completion.json)")
    write_json(args.output / "summary.json", {
        "model_id": manifest["model_id"], "off_diagonal_only": True, "by_target": summary,
        "overall": {metric: float(np.mean([row[metric] for row in summary])) for metric in ("mean_auroc", "mean_auprc")} if summary else {},
    })
    write_json(args.output / "completion.json", {
        "status": "completed", "model_id": manifest["model_id"], "subset_demo": manifest["subset_demo"],
        "species": len(species), "cells": len(species) ** 2 if args.cross_domain else len(species),
        "cross_domain": args.cross_domain, "encoder_updates": 0,
        "diagonal_replay": "passed" if reference else "not_requested",
        "diagonal_tolerance": args.diagonal_tolerance, "max_diagonal_error": max(diagonal_errors, default=0.0),
        "diagonal_reference_sha256": sha256(args.diagonal_reference) if reference else None,
        "embeddings_manifest_sha256": sha256(args.embeddings / "manifest.json"),
        "source_sha256": sha256(Path(__file__)), "numpy": np.__version__,
        "outputs_sha256": {name: sha256(args.output / name) for name in ("metrics.tsv", "summary.json")},
    })
    print(f"Completed {manifest['model_id']}: {len(species)} species, cross_domain={args.cross_domain}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    encoder = commands.add_parser("encode", help="Extract frozen first-token embeddings from local checkpoints")
    encoder.add_argument("--data-root", type=Path, required=True)
    encoder.add_argument("--checkpoint", type=Path, required=True)
    encoder.add_argument("--model-id", choices=MODEL_IDS, required=True)
    encoder.add_argument("--output", type=Path, required=True)
    encoder.add_argument("--device", default="cpu")
    encoder.add_argument("--batch-size", type=int, help="Default: Evo 32, NT-2.5B 64, NT-v2 128, others 256")
    encoder.add_argument("--allow-subset", action="store_true")
    encoder.add_argument("--trust-checkpoint-code", action="store_true")
    encoder.set_defaults(action=encode)
    scorer = commands.add_parser("score", help="Score same-dataset or all source-to-target prototype pairs")
    scorer.add_argument("--embeddings", type=Path, required=True)
    scorer.add_argument("--output", type=Path, required=True)
    scorer.add_argument("--cross-domain", action="store_true")
    scorer.add_argument("--diagonal-reference", type=Path)
    scorer.add_argument("--diagonal-tolerance", type=float, default=0.005)
    scorer.set_defaults(action=score)
    args = parser.parse_args()
    args.action(args)


if __name__ == "__main__":
    main()
