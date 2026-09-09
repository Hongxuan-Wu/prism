# Additional evaluation experiments

This directory adds evaluation-only entry points to the existing PRISM demo.
The existing framework, dependencies, and model architectures are retained;
see [downstream compatibility notes](../1train/README.md) and
[legacy selection fixes](../2select/README.md) for targeted maintenance updates.
No pretraining, fine-tuning, optimizer, model-generation training,
server orchestration, model weights, or experiment datasets are included here.

| Entry point | Purpose |
| --- | --- |
| `prototype.py encode` | Frozen-checkpoint inference on the original reference/test split |
| `prototype.py score` | Within-dataset reference prototypes, or cross-dataset transfer |
| `blast.py run` | Fixed-parameter, explicitly scoped local BLAST search |
| `blast.py summarize` | Receipt validation, best hits, threshold categories, and denominators |
| `check_evaluation.py` | Offline synthetic checks without a model/database download |
| `check_legacy.py` | CPU checks for legacy label/checkpoint compatibility and selection edge cases |

Use the Python environment described in the root README. Prototype scoring uses
NumPy and scikit-learn; encoding additionally uses PyTorch and Transformers.
The BLAST wrapper uses only the Python standard library and an existing BLAST+
2.17.0 installation. Model-specific checkpoint dependencies must already be
installed in a compatible isolated environment. Nothing is installed or fetched
automatically. Run commands below from the repository root.

## 1. Frozen reference-set prototypes

Supply the original persisted random 8:2 split:

```text
pdd/
  Train_catATG/<species>_catATG_train.csv
  Test_catATG/<species>_catATG_test.csv
```

Each CSV has exactly `sequences,labels` columns: the original 801-bp IUPAC DNA
sequence and a binary label (0/1). Both classes must occur in each file. Preserve
the original sequence including its shared reporter context; this entry point
does not remove it or create a new split. By default, the checker requires all
28 species and the original totals of 206,137 reference and 51,551 test rows.
`--allow-subset` permits a smaller demo and marks its output as such. Matching
counts are a structural check, not proof that an arbitrary dataset is the
original release. Use the supplied manuscript data, not a newly randomized or
cluster-disjoint split. Input file hashes are saved with the embeddings.

```bash
python 3evaluate/prototype.py encode \
  --data-root /path/to/pdd \
  --checkpoint /path/to/local/PRISM-M1 \
  --model-id m1 --device cuda:0 \
  --trust-checkpoint-code \
  --output /path/to/evaluation/m1_embeddings

python 3evaluate/prototype.py score \
  --embeddings /path/to/evaluation/m1_embeddings \
  --output /path/to/evaluation/m1_within
```

Checkpoint code/weight loading can execute code. Review and obtain your local
checkpoint from a trusted source before using `--trust-checkpoint-code`.
Only local checkpoint files are loaded. The encoder is frozen (`eval()`, no
gradients), with first-token pooling, 100-token Transformer truncation, and
400-character Evo truncation, as in the original evaluation. Tokenizer-added
special tokens and its padding behavior are retained. CUDA uses FP16 autocast
for Transformers and BF16 for Evo; CPU inference does not use autocast.

Supported model IDs are `dnabert2`, `ntv2_500m`, `nt_2.5b`, `evo8k`, `m1`, `m2`,
`m8`, and `random_genome`. Each invocation evaluates one supplied checkpoint.
Evo requires the compatible local `configuration_hyena.py` and `modeling_hyena.py`
checkpoint implementation. Default batch sizes are 32/64/128 for Evo/NT-2.5B/NT-v2,
and 256 otherwise. Reduce `--batch-size` if necessary; changing batch size,
precision, software, or checkpoints may affect numerical replay. Actual runtime
versions, checkpoint file hashes, and preprocessing settings are recorded.
The random-genome model is a control matched except for training precision;
it should not be described as a perfectly precision-matched causal control.

For each reference class, average L2-normalized reference embeddings and
L2-normalize that mean. The test score is positive-prototype cosine similarity
minus negative-prototype cosine similarity. Test labels are not arguments to
score construction; they are used only to compute metrics afterward.

Primary metrics are AUROC and **trapezoidal AUPRC** (`auc(recall, precision)`),
not average precision. Accuracy and F1 use a strict `margin > 0` threshold (ties
are negative). Positive-only cosine AUROC/AUPRC are retained as compatibility
metrics. This uses labeled reference examples and is **not strict zero-shot**.

## 2. Cross-dataset prototype transfer

Reuse the same embeddings; no checkpoint update or additional encoder inference
is needed. Source A's original reference labels construct its prototypes;
these score target B's test sequences directly. No target labels construct or
tune a prototype. All source-target pairs, including the diagonal, are retained.

```bash
python 3evaluate/prototype.py score \
  --embeddings /path/to/evaluation/m1_embeddings \
  --cross-domain \
  --diagonal-reference /path/to/evaluation/m1_within/metrics.tsv \
  --output /path/to/evaluation/m1_cross
```

`metrics.tsv` contains six metrics for each pair: 784 cells per model for a full
28-species roster. `summary.json` excludes the 28 diagonal cells, averages the
27 foreign-source scores within each target, then averages targets equally.
Eight separately evaluated models therefore have 6,048 off-diagonal cells.
Cross-species AUROC/AUPRC are the primary readouts; a zero margin is not a
cross-species calibrated probability threshold. These comparisons are
descriptive, not independent-replicate tests or a universal model ranking.

The optional diagonal reference must cover all six metrics for every diagonal
cell of the current model. Its default absolute tolerance is 0.005, matching
the accepted mixed-precision replay criterion; a stricter tolerance may be
specified. Replaying a reference generated from the same cache tests scoring
consistency, not an independent encoder reproduction. Without a reference,
the receipt explicitly says `diagonal_replay: not_requested`. A failed gate
leaves no `completion.json`; intermediate metrics must not be treated as accepted.

This is prototype transfer, not regression training or cross-domain fine-tuning.
Those training pipelines are intentionally not part of this public increment.

## 3. BLAST similarity audit

Supply a pre-verified local nucleotide database and explicitly name its release
or snapshot. Do not assume the rolling `core_nt` download URL is a version pin.
The database label is supplied by the operator; this wrapper does not download,
build, or independently checksum the entire database. For exact result replay,
use the same verified database contents, tool build, and query FASTA.

```bash
python 3evaluate/blast.py run \
  --query /path/to/unique_promoters.fasta \
  --database /path/to/verified_database/core_nt \
  --database-id core_nt-2026-07-18_snapshot-2026-07-21-01-05-02 \
  --blastn /path/to/ncbi-blast-2.17.0+/bin/blastn \
  --taxid 561 --threads 8 --timeout-seconds 3600 \
  --output /path/to/evaluation/blast_escherichia

python 3evaluate/blast.py summarize \
  --run-dir /path/to/evaluation/blast_escherichia \
  --output /path/to/evaluation/blast_escherichia_summary
```

Genus taxids are Escherichia **561**, Streptomyces **1883**, and Vibrio **662**.
Prepare the matching genus-specific queries and run each scope separately.
Use `--unrestricted` instead of `--taxid` only for an explicitly intended
unrestricted search; the wrapper does not silently broaden the search scope.
Split large workloads into explicit user-managed FASTA batches if needed.

Fixed parameters are `-task blastn -word_size 11 -evalue 10 -dust no
-soft_masking false -strand both -max_target_seqs 10 -max_hsps 1`. The 16 output
fields are defined in `blast.py`; ordinary default 12-column output is not
accepted. A successful exit, valid rows, query coverage counts, and raw/query
SHA256 hashes are required before a completed receipt is written. Failed or
timed-out runs retain diagnostics and cannot be summarized as no-hit results.

The best reported hit is chosen by descending bitscore, ascending E-value,
descending identity, descending query coverage, then accession. Categories are
full-query exact; identity >=95%, >=90%, or >=80% with query coverage >=80%;
other reported hit; and no reported hit. Full-query exact does **not** require
the entire database subject to be the same length. Both mutually exclusive
categories and cumulative thresholds are reported. A high-identity hit with
insufficient query coverage remains an **other reported hit**, not no hit.

Every unique FASTA identifier is counted, including those without reported hits.
The script does not deduplicate sequences automatically. To reproduce a
unique-sequence denominator, supply an already deduplicated FASTA. An optional
`summarize --weights /path/to/weights.tsv` accepts exactly `query_id` and `weight`
columns, with each query once and a positive integer multiplicity. This adds
row-weighted counts without silently changing the query-ID denominator.

This entry point implements **core_nt BLAST**, not the separate exact matching
plus MMseqs2 searches against the generation-reference corpus or EC. The latter
require both query and target coverage >=80%; core_nt requires query coverage
only. Do not relabel MMseqs2 output as BLAST. The manuscript's limited
unrestricted audit is not evidence of an exhaustive unrestricted search, and
absence of a reported hit is not proof of absolute biological novelty.

## Offline checks and scope of validation

```bash
python 3evaluate/check_evaluation.py --work-dir /path/to/fresh/offline_checks
```

The checks cover prototype geometry, label separation, input/cache validation,
within/cross summaries, diagonal-replay rejection, frozen first-token inference,
BLAST arguments, threshold boundaries, weighted denominators, and failed versus
successful no-hit runs. Synthetic files are retained for inspection. These are
CPU checks with mocked BLAST execution, not full checkpoint inference or a
search against core_nt. No supplied/private sequences are embedded in the tests.

All output directories must be fresh. Store results outside the source checkout;
only evaluation code and these instructions are distributed. Scoring,
first-token feature extraction, BLAST parameters, hit ordering, and categories
are adapted from the corresponding finalized PRISM evaluation implementations;
private run manifests, infrastructure details, training dependencies, and Git
history are intentionally not required by these public entry points.
