#!/usr/bin/env python3
"""Run and summarize a local, explicitly scoped BLAST nucleotide similarity audit."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import subprocess
from collections import Counter
from pathlib import Path

RAW_FIELDS = (
    "qseqid", "saccver", "pident", "qcovhsp", "length", "mismatch", "gapopen",
    "qstart", "qend", "sstart", "send", "evalue", "bitscore", "qlen", "slen", "staxids",
)
CATEGORIES = (
    "full_query_exact", "identity_ge95_qcov_ge80", "identity_ge90_qcov_ge80",
    "identity_ge80_qcov_ge80", "other_reported_hit", "no_reported_hit",
)
CUMULATIVE = (
    "full_query_exact", "identity_ge95_qcov_ge80_including_exact",
    "identity_ge90_qcov_ge80_including_higher", "identity_ge80_qcov_ge80_including_higher",
    "any_reported_hit",
)
PARAMETERS = (
    "-task", "blastn", "-word_size", "11", "-evalue", "10", "-dust", "no",
    "-soft_masking", "false", "-strand", "both", "-max_target_seqs", "10", "-max_hsps", "1",
)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def read_fasta(path):
    records, name, sequence = {}, None, []

    def finish():
        if name is None:
            return
        text = "".join(sequence).upper()
        if not text or set(text) - set("ACGT") or name in records:
            raise ValueError("FASTA must contain unique IDs and non-empty A/C/G/T sequences")
        records[name] = len(text)

    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                finish()
                fields = line[1:].split()
                if not fields:
                    raise ValueError("Empty FASTA identifier")
                name, sequence = fields[0], []
            elif name is None:
                raise ValueError("Sequence appears before a FASTA header")
            else:
                sequence.append(line)
    finish()
    if not records:
        raise ValueError("Query FASTA is empty")
    return records


def best_key(row):
    return (-float(row["bitscore"]), float(row["evalue"]), -float(row["pident"]),
            -float(row["qcovhsp"]), row["saccver"])


def blast_category(row):
    if row is None:
        return "no_reported_hit"
    if (float(row["pident"]) == 100.0 and float(row["qcovhsp"]) == 100.0
            and int(row["length"]) == int(row["qlen"])
            and int(row["mismatch"]) == 0 and int(row["gapopen"]) == 0):
        return "full_query_exact"
    if float(row["qcovhsp"]) >= 80.0:
        for threshold in (95, 90, 80):
            if float(row["pident"]) >= threshold:
                return f"identity_ge{threshold}_qcov_ge80"
    return "other_reported_hit"


def read_hits(path, queries):
    best, count = {}, 0
    with path.open(newline="", encoding="utf-8") as handle:
        for values in csv.reader(handle, delimiter="\t"):
            if len(values) != len(RAW_FIELDS):
                raise ValueError("BLAST output does not match the required 16-column format")
            row = dict(zip(RAW_FIELDS, values))
            query = row["qseqid"]
            if query not in queries or int(row["qlen"]) != queries[query]:
                raise ValueError("Unknown query or inconsistent query length")
            for field in ("pident", "qcovhsp", "evalue", "bitscore"):
                number = float(row[field])
                if not math.isfinite(number) or number < 0:
                    raise ValueError("Invalid BLAST numeric field")
            if float(row["pident"]) > 100 or float(row["qcovhsp"]) > 100:
                raise ValueError("BLAST percentages must be in [0, 100]")
            for field in ("length", "qlen", "slen", "qstart", "qend", "sstart", "send"):
                if int(row[field]) < 1:
                    raise ValueError("Invalid BLAST length or coordinate")
            if (max(int(row["qstart"]), int(row["qend"])) > int(row["qlen"])
                    or max(int(row["sstart"]), int(row["send"])) > int(row["slen"])
                    or min(int(row["mismatch"]), int(row["gapopen"])) < 0):
                raise ValueError("Invalid BLAST coordinate/count")
            count += 1
            if query not in best or best_key(row) < best_key(best[query]):
                best[query] = row
    return best, count


def blast_argv(binary, database, query, output, threads, taxid):
    argv = [str(binary), "-db", str(database), "-query", str(query), *PARAMETERS,
            "-num_threads", str(threads), "-outfmt", "6 " + " ".join(RAW_FIELDS), "-out", str(output)]
    if taxid is not None:
        argv.extend(["-taxids", str(taxid)])
    return argv


def run(args):
    if not 1 <= args.threads <= (os.cpu_count() or 1) or args.timeout_seconds < 1:
        raise ValueError("Specify valid positive CPU/time limits")
    if args.taxid is not None and args.taxid < 1:
        raise ValueError("Taxid must be a positive integer")
    if not args.database_id.strip():
        raise ValueError("A database release/snapshot identifier is required")
    queries = read_fasta(args.query)
    binary = args.blastn.resolve(strict=True)
    environment = os.environ.copy()
    environment.update({"BLAST_USAGE_REPORT": "false", "NCBI_DONT_USE_LOCAL_CONFIG": "true",
                        "NCBI_DONT_USE_NCBIRC": "true", "BLASTDB": str(args.database.resolve().parent)})
    version = subprocess.check_output([str(binary), "-version"], env=environment, text=True, timeout=30).splitlines()[0]
    if version.strip() != "blastn: 2.17.0+":
        raise ValueError("This protocol requires BLAST+ 2.17.0+")
    args.output.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(args.query, args.output / "queries.fasta")
    raw = args.output / "raw.tsv"
    argv = blast_argv(binary, args.database.resolve(), (args.output / "queries.fasta").resolve(),
                      raw.resolve(), args.threads, args.taxid)
    receipt = {
        "schema": "prism-public-blast/v1", "status": "running", "database_id": args.database_id,
        "taxid": args.taxid, "scope": "taxid_restricted" if args.taxid else "unrestricted",
        "query_sha256": sha256(args.output / "queries.fasta"), "query_count": len(queries),
        "blastn_version": version, "blastn_sha256": sha256(binary), "argv": argv,
        "threads": args.threads, "timeout_seconds": args.timeout_seconds, "parameters": list(PARAMETERS),
        "source_sha256": sha256(Path(__file__)),
    }
    try:
        with (args.output / "blast.log").open("x", encoding="utf-8") as log:
            subprocess.run(argv, stdout=log, stderr=subprocess.STDOUT, env=environment,
                           timeout=args.timeout_seconds, check=True)
        best, raw_rows = read_hits(raw, queries)
        receipt.update(status="completed", raw_sha256=sha256(raw), raw_rows=raw_rows, hit_queries=len(best))
    except Exception as error:
        receipt.update(status="failed", error_type=type(error).__name__)
        write_json(args.output / "receipt.json", receipt)
        raise
    write_json(args.output / "receipt.json", receipt)
    print(f"Completed BLAST: {len(queries)} query IDs, {len(best)} with reported hits")


def summarize(args):
    root = args.run_dir
    receipt = json.loads((root / "receipt.json").read_text(encoding="utf-8"))
    if (receipt.get("schema") != "prism-public-blast/v1" or receipt.get("status") != "completed"
            or receipt.get("parameters") != list(PARAMETERS)
            or receipt.get("blastn_version") != "blastn: 2.17.0+"
            or receipt.get("scope") != ("taxid_restricted" if receipt.get("taxid") else "unrestricted")):
        raise ValueError("A completed fixed-protocol BLAST receipt is required")
    argv = receipt.get("argv")
    if not isinstance(argv, list) or not argv or not all(isinstance(value, str) for value in argv):
        raise ValueError("BLAST command is missing from receipt")
    try:
        expected = blast_argv(argv[0], argv[argv.index("-db") + 1], argv[argv.index("-query") + 1],
                              argv[argv.index("-out") + 1], receipt["threads"], receipt["taxid"])
    except (IndexError, ValueError, KeyError) as error:
        raise ValueError("Malformed BLAST command receipt") from error
    if argv != expected:
        raise ValueError("Recorded BLAST command differs from the fixed protocol")
    for name, key in (("queries.fasta", "query_sha256"), ("raw.tsv", "raw_sha256")):
        if sha256(root / name) != receipt[key]:
            raise ValueError(f"BLAST artifact checksum mismatch: {name}")
    queries = read_fasta(root / "queries.fasta")
    best, raw_rows = read_hits(root / "raw.tsv", queries)
    if (len(queries) != receipt["query_count"] or raw_rows != receipt["raw_rows"]
            or len(best) != receipt["hit_queries"]):
        raise ValueError("BLAST receipt coverage mismatch")
    weights = dict.fromkeys(queries, 1)
    if args.weights:
        weights = {}
        with args.weights.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            if reader.fieldnames != ["query_id", "weight"]:
                raise ValueError("Weights TSV must have query_id and weight columns")
            for row in reader:
                query, weight = row["query_id"], int(row["weight"])
                if query in weights or query not in queries or weight < 1:
                    raise ValueError("Invalid or duplicate query weight")
                weights[query] = weight
        if set(weights) != set(queries):
            raise ValueError("Weights must cover every query exactly once")
    args.output.mkdir(parents=True, exist_ok=False)
    unique, weighted = Counter(), Counter()
    with (args.output / "best_hits.tsv").open("x", newline="", encoding="utf-8") as handle:
        fields = (*RAW_FIELDS, "category", "weight")
        writer = csv.DictWriter(handle, fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for query, length in queries.items():
            row = best.get(query)
            category = blast_category(row)
            unique[category] += 1
            weighted[category] += weights[query]
            writer.writerow({**(row or {"qseqid": query, "qlen": length}),
                             "category": category, "weight": weights[query]})
    with (args.output / "summary.tsv").open("x", newline="", encoding="utf-8") as handle:
        fields = ("denominator", "total", "summary_type", "category", "count", "fraction")
        writer = csv.DictWriter(handle, fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for name, counts in (("query_ids", unique), ("weighted_rows", weighted)):
            total = sum(counts.values())
            for category in CATEGORIES:
                writer.writerow(dict(denominator=name, total=total, summary_type="exclusive",
                                     category=category, count=counts[category], fraction=counts[category] / total))
            for index, category in enumerate(CUMULATIVE):
                count = sum(counts[c] for c in CATEGORIES[:index + 1])
                writer.writerow(dict(denominator=name, total=total, summary_type="cumulative",
                                     category=category, count=count, fraction=count / total))
    write_json(args.output / "completion.json", {
        "status": "completed", "query_ids": len(queries), "weighted_rows": sum(weights.values()),
        "weights_sha256": sha256(args.weights) if args.weights else None,
        "receipt_sha256": sha256(root / "receipt.json"), "source_sha256": sha256(Path(__file__)),
        "database_id": receipt["database_id"], "scope": receipt["scope"], "taxid": receipt["taxid"],
        "outputs_sha256": {name: sha256(args.output / name) for name in ("best_hits.tsv", "summary.tsv")},
    })
    print(f"Summarized {len(queries)} query IDs; scope={receipt['scope']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    runner = commands.add_parser("run", help="Run fixed-parameter BLAST against an existing local database")
    runner.add_argument("--query", type=Path, required=True)
    runner.add_argument("--database", type=Path, required=True)
    runner.add_argument("--database-id", required=True, help="Verified release/snapshot identity, not just core_nt")
    runner.add_argument("--blastn", type=Path, required=True)
    runner.add_argument("--output", type=Path, required=True)
    runner.add_argument("--threads", type=int, default=8)
    runner.add_argument("--timeout-seconds", type=int, default=3600)
    scope = runner.add_mutually_exclusive_group(required=True)
    scope.add_argument("--taxid", type=int)
    scope.add_argument("--unrestricted", action="store_true")
    runner.set_defaults(action=run)
    summary = commands.add_parser("summarize", help="Verify receipts and categorize every query, including no-hit queries")
    summary.add_argument("--run-dir", type=Path, required=True)
    summary.add_argument("--weights", type=Path)
    summary.add_argument("--output", type=Path, required=True)
    summary.set_defaults(action=summarize)
    args = parser.parse_args()
    args.action(args)


if __name__ == "__main__":
    main()
