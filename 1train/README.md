# Existing downstream evaluation entry points

The directory name is retained for compatibility. This update adds no training
loop, optimizer, or pretraining code. Existing model architectures, input splits,
tokenization, pooling options, and prediction filenames are preserved.

## Configure a downstream checkpoint

Use the isolated environment and dataset download described in the
[root README](../README.md). Run commands from the repository root.

In the selected task's `config.py`, set `config['root']` to the extracted data
directory, including its trailing slash, **before** the derived path settings.
Set `config['mode'] = 'test'` before the mode-dependent settings. For transcript
classification, also set `config['num_class']` to the checkpoint's class count.
Do not change the dataset, class thresholds, reporter context, or pooling option
when reproducing an existing result.

For a separately supplied checkpoint, place these overrides at the **end** of
that task's configuration, after the default paths have been constructed:

```python
config['pretrained_path'] = '/path/to/trusted/local/backbone-and-tokenizer/'
config['checkpoint_dir'] = '/path/to/matching/downstream-task-checkpoint/'
```

```bash
python 1train/transcript_level_cls/main.py \
  --config_file 1train/transcript_level_cls/config.py
```

The downstream directory must contain `last.pth`, a complete state dictionary
for the selected task model, not just its pretrained encoder. Checkpoint loading
may execute code; only use trusted files. No weights or data are included by
this update, and no checkpoint is trained automatically.

| Evaluation | Required checkpoint |
| --- | --- |
| Existing M1 demo | Its published backbone/tokenizer and matching task-specific `last.pth` |
| M2, M8, or random-genome supervised evaluation | A compatible 768-dimensional backbone/tokenizer and the matching task-specific `last.pth`, supplied separately |
| Frozen M1/M2/M8/random-genome or NT/Evo prototypes | A supported local encoder checkpoint; use [3evaluate](../3evaluate/README.md), not a legacy task head |

Changing only the encoder path is not a valid new-model supervised benchmark:
the loaded task state also contains backbone weights. NT/Evo task checkpoints
are not interchangeable with the existing 768-dimensional demo heads. The
generation and component-analysis models likewise require their own matching
task checkpoints. The random-genome control is matched except for training
precision, not a perfectly precision-matched causal control.

## Compatibility fixes and limits

- Transcript-classification labels are validated against the configured class
  count. Binary losses receive floating-point targets; multiclass losses receive
  integer targets, even when a test subset omits some classes.
- A binary subset containing only one class has undefined AUROC (`NaN`) and
  empty ROC arrays. It is not assigned an artificial score of zero or one.
- Multiclass checkpoint keys beginning with `classifier.` are mapped to the
  existing `classification.` head name, including an existing `module.` prefix.
  Legacy names remain supported. Conflicting aliases fail; state dictionaries
  still load strictly, including tensor dimensions and all other keys. Match
  the checkpoint's single-device/DataParallel wrapper to the existing entry point.
- These fixes do not change the normal complete-dataset metric definitions or
  retrain any model. GPU inference and manuscript-checkpoint replay require the
  corresponding environment, data, and weights; synthetic checks do not prove
  full experimental reproduction.

For generation followed by prediction, all three scoring tasks must read the
**same unchanged** `gen_seqs.csv` in the same order. Keep their four input/output
CSV files together in a run-specific prediction directory. See
[selection inputs and checks](../2select/README.md) before running BLAST.

Run the small CPU regression check without a checkpoint or BLAST database:

```bash
python 3evaluate/check_legacy.py --work-dir /path/to/fresh/legacy_checks
```
