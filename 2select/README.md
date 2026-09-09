# Legacy single-genome selection

This entry point preserves the original ranking and E. coli K-12 MG1655
reference-genome selection workflow. It is **not** the genus-matched `core_nt`
audit. Use [3evaluate/blast.py](../3evaluate/README.md#3-blast-similarity-audit)
for the separately defined BLAST+ 2.17.0 audit and its coverage categories.

## Inputs and command

Use the Python environment described in the [root README](../README.md) and an
existing local BLAST+ installation. The legacy example uses BLAST+ 2.16.0; the
script retains BLAST's default search parameters and `-outfmt 6`. Record the
actual tool version if comparing runs; defaults and results can differ across
versions. No tool or database is downloaded automatically.

The prediction directory must contain four headerless, single-column CSV files:

| File | Content |
| --- | --- |
| `gen_seqs.csv` | Generated DNA sequences, in their original order |
| `authenticity_cls.csv` | Authenticity scores for those exact rows |
| `transcript_level_cls2.csv` | Binary transcript-level scores for those exact rows |
| `transcript_level_cls4.csv` | Four-class predictions for those exact rows |

```bash
python 2select/main.py \
  --blast-dir /path/to/ncbi-blast-2.16.0+ \
  --predict-dir /path/to/predictions \
  --output-dir /path/to/fresh/selection
```

`--blast-dir` names the installation containing `bin/`; omit it to use
`makeblastdb` and `blastn` already on `PATH`. `--root-dir` defaults to this
repository, which supplies `2select/fasta/GCF_000005845.2_ASM584v2_genomic.fna`.
Input and output directories default to `results/predicts` and `results/select`
under that root. Output directories must be fresh; an existing directory is
never reused by the command. Database index files are written inside the output
directory, not beside the reference FASTA. Paths containing spaces are supported.

## Preserved behavior and explicit edge cases

- Ranking remains descending authenticity, binary transcript score, then
  four-class prediction. The original zero-based input-row index is retained
  through sorting and used as the FASTA query identifier.
- Input tables must be nonempty, have equal row counts and one column each,
  contain no missing values, and have finite numeric predictions. Because the
  legacy prediction files have no identifiers, equal counts **cannot detect
  same-length files from a different run or a different row order**. Regenerate
  all three prediction files from the same unchanged sequence file.
- Sequences are normalized to uppercase and must contain only A/C/G/T. The
  reporter marker remains exactly `ATGGTGAGCAAGGGCGAGGA`, and the first occurrence
  defines the cut. If absent, the entire sequence is retained with a warning
  and `reporter_boundary_found=False`; the last base is not discarded. A marker
  at the start, leaving an empty promoter, or invalid DNA causes an error.
  Mask tokens are not silently removed. Inspect missing-boundary rows before
  interpreting them as promoter-only searches.
- The legacy filter still keeps the first reported alignment per query; it is
  not the deterministic best-hit/category procedure in `3evaluate`.
- A successful BLAST run with zero hits produces a header-only hit table and a
  merged table retaining **every** query. `blast_hit_reported=False` identifies
  no-hit rows; alignment fields remain missing rather than being set to zero.
  Unknown or duplicate query IDs are rejected during merging. A subprocess
  failure stops the command; it is not interpreted as a successful no-hit run.

The final output is `predicts_sorted_cut_blast_merge.csv`. Missing-boundary and
no-hit handling repair edge cases; they do not establish biological novelty or
convert the legacy reference-genome search into a `core_nt` experiment.

```bash
python 3evaluate/check_legacy.py --work-dir /path/to/fresh/legacy_checks
```

This check uses synthetic inputs, CPU-only classifier stubs, and mocked BLAST
execution. It does not rerun manuscript checkpoints, GPU inference, or BLAST.
