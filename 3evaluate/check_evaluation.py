#!/usr/bin/env python3
"""Offline synthetic checks; outputs are retained in a fresh user-specified directory."""

import argparse
import csv
import inspect
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

import blast
import prototype as proto


def rejected(function, *args):
    try:
        function(*args)
    except (ValueError, FileExistsError):
        return
    raise AssertionError("Invalid input was accepted")


def cache_fixture(root):
    root.mkdir()
    features = np.asarray([[1., 0., 0.], [0.8, 0.2, 0.], [0., 1., 0.], [0.2, 0.8, 0.]], dtype=np.float32)
    labels = np.asarray([0, 0, 1, 1], dtype=np.int8)
    records = []
    for species in ("species_a", "species_b"):
        for split in ("train", "test"):
            name = f"{species}_{split}.npz"
            np.savez(root / name, features=features, labels=labels)
            records.append(dict(species=species, split=split, file=name, rows=4,
                                sha256=proto.sha256(root / name), input_sha256="synthetic_fixture"))
    manifest = dict(schema="prism-public-embeddings/v1", status="completed", model_id="dnabert2",
                    split_family="original_random_8_2", subset_demo=True, encoder_updates=0,
                    pooling="first_token", transformer_max_tokens=100, evo_max_characters=400, files=records)
    proto.write_json(root / "manifest.json", manifest)
    return features, labels


def check_prototypes(root):
    cache = root / "embeddings"
    features, labels = cache_fixture(cache)
    prototypes = proto.class_prototypes(features, labels)
    margin, positive, negative = proto.score_features(prototypes, features)
    assert np.allclose(margin, positive - negative)
    assert proto.metric_values(labels, margin, positive)["auroc"] == 1.0
    assert "test_labels" not in inspect.signature(proto.score_features).parameters
    rejected(proto.class_prototypes, features, [0, 0, 0, 0])
    rejected(proto.class_prototypes, features, [0, 0, 257, 257])
    rejected(proto.normalized_features, np.zeros((2, 3)))
    rejected(proto.normalized_features, np.asarray([[np.nan, 1]]))
    rejected(proto.score_features, prototypes, np.ones((2, 4)))
    diagonal = root / "diagonal"
    args = SimpleNamespace(embeddings=cache, output=diagonal, cross_domain=False,
                           diagonal_reference=None, diagonal_tolerance=0.005)
    proto.score(args)
    rejected(proto.score, args)
    cross = root / "cross"
    args = SimpleNamespace(embeddings=cache, output=cross, cross_domain=True,
                           diagonal_reference=diagonal / "metrics.tsv", diagonal_tolerance=0.005)
    proto.score(args)
    completion = json.loads((cross / "completion.json").read_text())
    assert completion["cells"] == 4 and completion["diagonal_replay"] == "passed"
    with (cross / "metrics.tsv").open() as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    assert len(rows) == 24 and sum(r["evaluation_scope"] == "off_diagonal" for r in rows) == 12
    assert json.loads((cross / "summary.json").read_text())["overall"]["mean_auroc"] == 1.0
    with (diagonal / "metrics.tsv").open() as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        fields, reference = reader.fieldnames, list(reader)
    reference[0]["value"] = "0"
    bad_reference = root / "bad_reference.tsv"
    with bad_reference.open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(reference)
    args.diagonal_reference, args.output = bad_reference, root / "failed_diagonal"
    rejected(proto.score, args)
    assert not (args.output / "completion.json").exists()
    with (cache / "species_a_test.npz").open("ab") as handle:
        handle.write(b"deliberate synthetic corruption")
    rejected(proto.load_embeddings, cache)
    csv_path = root / "bad_sequences.csv"
    csv_path.write_text("sequences,labels\nACGT,0\nACGT,1\n")
    rejected(proto.read_sequences, csv_path)


def hit(query, identity, coverage, length):
    return dict(qseqid=query, saccver="synthetic.1", pident=str(identity), qcovhsp=str(coverage),
                length=str(length), mismatch="0" if identity == 100 else "1", gapopen="0",
                qstart="1", qend=str(length), sstart="1", send=str(length), evalue="1e-5",
                bitscore="100", qlen="100", slen="100", staxids="561")


def check_blast(root):
    query = root / "queries.fasta"
    query.write_text("".join(f">q{i}\n{'ACGT' * 25}\n" for i in range(6)))
    binary = root / "mock_binary.txt"
    binary.write_text("Synthetic placeholder; never executed.\n")
    rows = [hit("q0", 100, 100, 100), hit("q1", 95, 80, 80), hit("q2", 90, 80, 80),
            hit("q3", 80, 80, 80), hit("q4", 100, 79, 79)]
    assert [blast.blast_category(row) for row in rows] == list(blast.CATEGORIES[:-1])
    assert blast.blast_category(None) == "no_reported_hit"

    def fake_run(argv, **kwargs):
        assert "shell" not in kwargs
        assert argv[argv.index("-taxids") + 1] == "561"
        assert argv[argv.index("-word_size") + 1] == "11"
        assert kwargs["env"]["BLAST_USAGE_REPORT"] == "false"
        with Path(argv[argv.index("-out") + 1]).open("x", newline="") as handle:
            writer = csv.DictWriter(handle, blast.RAW_FIELDS, delimiter="\t", lineterminator="\n")
            writer.writerows(rows)
        return subprocess.CompletedProcess(argv, 0)

    args = SimpleNamespace(query=query, blastn=binary, database=root / "mock_db", database_id="synthetic-test-only",
                           output=root / "blast_run", taxid=561, threads=1, timeout_seconds=5)
    with patch.object(blast.subprocess, "check_output", return_value="blastn: 2.17.0+\n"), \
            patch.object(blast.subprocess, "run", side_effect=fake_run):
        blast.run(args)
    summary = SimpleNamespace(run_dir=args.output, weights=None, output=root / "blast_summary")
    blast.summarize(summary)
    with (summary.output / "best_hits.tsv").open() as handle:
        reported = list(csv.DictReader(handle, delimiter="\t"))
    assert len(reported) == 6 and reported[-1]["category"] == "no_reported_hit"
    with (summary.output / "summary.tsv").open() as handle:
        summaries = list(csv.DictReader(handle, delimiter="\t"))
    assert len(summaries) == 22
    assert sum(int(r["count"]) for r in summaries if r["denominator"] == "query_ids" and r["summary_type"] == "exclusive") == 6
    weights = root / "weights.tsv"
    weights.write_text("query_id\tweight\n" + "".join(f"q{i}\t{3 if i == 0 else 1}\n" for i in range(6)))
    summary.weights, summary.output = weights, root / "weighted_summary"
    blast.summarize(summary)
    assert json.loads((summary.output / "completion.json").read_text())["weighted_rows"] == 8
    rows.clear()
    args.output = root / "no_hit_run"
    with patch.object(blast.subprocess, "check_output", return_value="blastn: 2.17.0+\n"), \
            patch.object(blast.subprocess, "run", side_effect=fake_run):
        blast.run(args)
    summary.run_dir, summary.output = args.output, root / "no_hit_summary"
    blast.summarize(summary)
    with (summary.output / "best_hits.tsv").open() as handle:
        assert all(row["category"] == "no_reported_hit" for row in csv.DictReader(handle, delimiter="\t"))
    receipt_path = args.output / "receipt.json"
    receipt = json.loads(receipt_path.read_text())
    receipt["argv"][receipt["argv"].index("-word_size") + 1] = "28"
    receipt_path.write_text(json.dumps(receipt))
    summary.output = root / "bad_command_summary"
    rejected(blast.summarize, summary)
    args.output = root / "failed_run"
    with patch.object(blast.subprocess, "check_output", return_value="blastn: 2.17.0+\n"), \
            patch.object(blast.subprocess, "run", side_effect=subprocess.CalledProcessError(1, ["mock"])):
        try:
            blast.run(args)
        except subprocess.CalledProcessError:
            pass
        else:
            raise AssertionError("Failed BLAST execution was accepted")
    assert json.loads((args.output / "receipt.json").read_text())["status"] == "failed"
    summary.run_dir, summary.output = args.output, root / "failed_summary"
    rejected(blast.summarize, summary)
    duplicate = root / "duplicate.fasta"
    duplicate.write_text(">same\nACGT\n>same\nACGT\n")
    rejected(blast.read_fasta, duplicate)
    nan_hit = hit("q0", "nan", 100, 100)
    invalid = root / "invalid_hit.tsv"
    invalid.write_text("\t".join(nan_hit[name] for name in blast.RAW_FIELDS) + "\n")
    rejected(blast.read_hits, invalid, {"q0": 100})


def check_encoder():
    import torch
    import frozen_encoder as encoder

    rejected(encoder.load_feature_model, "dnabert2", Path("not_loaded"), torch.device("cpu"))

    class HiddenModel(torch.nn.Module):
        def forward(self, input_ids, **kwargs):
            assert not torch.is_grad_enabled()
            hidden = input_ids.float().unsqueeze(-1).repeat(1, 1, 3)
            return SimpleNamespace(last_hidden_state=hidden)

    def tokenize(**kwargs):
        assert kwargs["max_length"] == 100 and kwargs["truncation"] is True
        assert kwargs["padding"] is True
        return {"input_ids": torch.tensor([[2, 9]] * len(kwargs["text"])),
                "attention_mask": torch.ones((len(kwargs["text"]), 2), dtype=torch.long)}

    model = HiddenModel().eval()
    features = encoder.extract_features("dnabert2", model, tokenize, ["A" * 801] * 3, 2, torch.device("cpu"))
    assert features.shape == (3, 3) and np.all(features == 2)
    evo = encoder.extract_features("evo8k", model, None, ["A" * 801], 1, torch.device("cpu"))
    assert evo.shape == (1, 3) and np.all(evo == ord("A"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, required=True)
    args = parser.parse_args()
    args.work_dir.mkdir(parents=True, exist_ok=False)
    check_prototypes(args.work_dir)
    check_blast(args.work_dir)
    check_encoder()
    proto.write_json(args.work_dir / "checks.json", {
        "status": "passed", "checks": ["prototype_geometry", "label_separation", "cache_integrity",
        "within_and_cross_cli_logic", "diagonal_rejection", "blast_command_and_categories",
        "query_coverage_and_weights", "failed_vs_no_hit", "frozen_inference_and_first_token"],
        "real_checkpoint_inference": False, "real_blast_search": False,
    })
    print("All offline synthetic checks passed; no checkpoint or database was required.")


if __name__ == "__main__":
    main()
