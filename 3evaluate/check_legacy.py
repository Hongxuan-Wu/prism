#!/usr/bin/env python3
"""CPU regression checks for legacy evaluation; retain synthetic artifacts in a fresh directory."""

import argparse
import importlib.util
import io
import json
import logging
import subprocess
import sys
import warnings
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch


REPO = Path(__file__).resolve().parents[1]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def rejected(error, function, *args):
    try:
        function(*args)
    except error:
        return
    raise AssertionError('Invalid input was accepted')


def check_classification(root):
    directory = REPO / '1train' / 'transcript_level_cls'
    data = load_module('legacy_data', directory / 'dataloader.py')
    model = load_module('legacy_model', directory / 'model.py')
    utils = load_module('legacy_utils', directory / 'utils.py')
    with patch.dict(sys.modules, {'dataloader': data, 'model': model, 'utils': utils}):
        trainer = load_module('legacy_trainer', directory / 'trainer.py')

    class Encoder(torch.nn.Module):
        def forward(self, input_ids, **kwargs):
            assert not torch.is_grad_enabled()
            return (input_ids.float().unsqueeze(-1).expand(-1, -1, 768),)

    def tokenizer(text, **kwargs):
        assert kwargs['max_length'] == 100
        return {'input_ids': torch.ones((len(text), 4), dtype=torch.long),
                'attention_mask': torch.ones((len(text), 4), dtype=torch.long)}

    cases = [(4, [0, 1]), (4, [0, 1, 2, 3]), (3, [0, 2]), (2, [0, 0]), (2, [1, 1]), (2, [0, 1])]
    for index, (num_class, labels) in enumerate(cases):
        config = dict(num_class=num_class, valid_batch_size=2, pretrained_path='synthetic',
                      checkpoint_dir='synthetic', output_type='pool', predict_dir=str(root / 'predicted'),
                      strong_weak=False)
        with patch.object(model.AutoConfig, 'from_pretrained', return_value=SimpleNamespace()), \
                patch.object(model.AutoModel, 'from_pretrained', return_value=Encoder()):
            torch.manual_seed(42)
            network = model.TranscriptLevelCls(config)
        frame = pd.DataFrame({'promoters': ['ACGT'] * len(labels), 'genes': ['ATG'] * len(labels),
                              'intensity_cls': labels})
        runner = trainer.Trainer.__new__(trainer.Trainer)
        runner.config, runner.model, runner.tokenizer, runner.df_val = config, network, tokenizer, frame
        runner.logger = logging.getLogger('legacy-check')
        runner.args = SimpleNamespace(num_workers=0, device='cpu', print_step=False, metrics_dir=str(root))
        runner.criterion = torch.nn.BCEWithLogitsLoss() if num_class == 2 else torch.nn.CrossEntropyLoss()
        state = network.state_dict()
        newer = {key.replace('classification.', 'classifier.', 1): value for key, value in state.items()}
        with patch.object(trainer.torch, 'load', return_value=state), \
                patch.object(trainer, 'autocast', return_value=nullcontext()):
            baseline_loss, baseline_metrics = runner.test()
        with patch.object(trainer.torch, 'load', return_value=newer), \
                patch.object(trainer, 'autocast', return_value=nullcontext()):
            loss, metrics = runner.test()
        assert np.isfinite(loss) and loss == baseline_loss
        assert metrics['accuracy'] == baseline_metrics['accuracy']
        if num_class == 2 and len(set(labels)) == 1:
            assert np.isnan(metrics['auc']) and len(metrics['fpr']) == 0

        if index == 0:
            runner.df_predict = pd.DataFrame(['ACGT', 'TGCA'])
            with patch.object(trainer.torch, 'load', return_value=newer):
                runner.predict()
            assert len(pd.read_csv(root / 'predicted' / 'transcript_level_cls4.csv', header=None)) == 2
            conflicting = dict(state, **newer)
            with patch.object(trainer.torch, 'load', return_value=conflicting):
                rejected(ValueError, trainer.load_checkpoint, 'unused', 'cpu', 4)
            parallel = {'module.' + key: value for key, value in newer.items()}
            with patch.object(trainer.torch, 'load', return_value=parallel):
                mapped = trainer.load_checkpoint('unused', 'cpu', 4)
                assert set(mapped) == {'module.' + key for key in state}
            wrong_shape = dict(state)
            wrong_shape['classification.dense.weight'] = torch.zeros(3, 768)
            rejected(RuntimeError, network.load_state_dict, wrong_shape)
            rejected(RuntimeError, network.load_state_dict, {})

    for invalid in ([], [-1], [2], [0.5], [np.nan]):
        frame = pd.DataFrame({'intensity_cls': invalid})
        rejected(ValueError, data.make_loader, runner.args, dict(num_class=2), frame, tokenizer)


def check_selection(root):
    # Import must not create an output directory or run external tools.
    with patch('os.makedirs', side_effect=AssertionError('Import attempted a filesystem write')):
        selection = load_module('legacy_selection', REPO / '2select' / 'main.py')
    inputs = root / 'prediction inputs'
    inputs.mkdir()
    marker = 'ATGGTGAGCAAGGGCGAGGA'
    pd.DataFrame(['ACGT' + marker, 'TTTT', 'GG' + marker]).to_csv(inputs / 'gen_seqs.csv', index=False, header=False)
    for name, values in [('authenticity_cls', [0.1, 0.9, 0.5]),
                         ('transcript_level_cls2', [0.2, 0.8, 0.6]), ('transcript_level_cls4', [1, 3, 2])]:
        pd.DataFrame(values).to_csv(inputs / (name + '.csv'), index=False, header=False)

    calls = []
    raw_hits = []

    def fake_run(argv, **kwargs):
        assert isinstance(argv, list) and 'shell' not in kwargs and kwargs['check'] is True
        assert kwargs['env']['BLAST_USAGE_REPORT'] == 'false'
        calls.append(argv)
        if Path(argv[0]).name == 'makeblastdb':
            assert Path(argv[argv.index('-out') + 1]).parent == Path(selection.select_dir)
            assert argv[argv.index('-dbtype') + 1] == 'nucl'
        else:
            assert Path(argv[0]).name == 'blastn'
            assert argv[argv.index('-outfmt') + 1] == '6'
            assert set(argv[1::2]) == {'-query', '-db', '-out', '-outfmt'}
            Path(argv[argv.index('-out') + 1]).write_text(''.join(raw_hits))
        return subprocess.CompletedProcess(argv, 0)

    for name in ('no hits', 'some hits'):
        output = root / name
        args = ['--blast-dir', str(root / 'mock blast installation'), '--predict-dir', str(inputs), '--output-dir', str(output)]
        if name == 'some hits':
            raw_hits.extend(['0\tsubject.first\t100\t4\t0\t0\t1\t4\t1\t4\t0.01\t20\n',
                             '0\tsubject.second\t100\t4\t0\t0\t1\t4\t1\t4\t0.001\t30\n'])
        with warnings.catch_warnings(record=True) as caught, patch.object(selection.subprocess, 'run', side_effect=fake_run):
            warnings.simplefilter('always')
            selection.main(args)
        assert any('lack the reporter boundary' in str(item.message) for item in caught)
        merged = pd.read_csv(output / 'predicts_sorted_cut_blast_merge.csv')
        assert merged['Unnamed: 0'].tolist() == [1, 2, 0]
        assert merged['sequence'].tolist() == ['TTTT', 'GG', 'ACGT']
        assert merged['reporter_boundary_found'].tolist() == [False, True, True]
        assert int(merged['blast_hit_reported'].sum()) == (1 if raw_hits else 0)
        assert (output / 'predicts_sorted_cut.fasta').read_text() == '>1\nTTTT\n>2\nGG\n>0\nACGT\n'
        if raw_hits:
            assert merged.loc[merged['blast_hit_reported'], 'Subject ID'].tolist() == ['subject.first']
        else:
            assert pd.read_csv(output / 'predicts_sorted_cut_blast.csv').shape == (0, 12)
            assert merged['% Identity'].isna().all()
        rejected(FileExistsError, selection.main, args)
    assert len(calls) == 4

    with patch.object(selection.subprocess, 'run', side_effect=subprocess.CalledProcessError(1, ['mock'])), \
            patch.object(selection, 'blast_filter', side_effect=AssertionError('Failed BLAST must not be summarized')), \
            warnings.catch_warnings():
        warnings.simplefilter('ignore')
        rejected(subprocess.CalledProcessError, selection.main,
                 ['--predict-dir', str(inputs), '--output-dir', str(root / 'failed run')])

    with patch.object(selection.pd, 'read_csv', return_value=pd.DataFrame({'sequence': [marker]})):
        rejected(ValueError, selection.cutATG)
    with patch.object(selection.pd, 'read_csv', return_value=pd.DataFrame({'sequence': ['AC[MASK]GT']})):
        rejected(ValueError, selection.cutATG)
    with patch('builtins.open', return_value=io.StringIO('malformed\trow\n')):
        rejected(ValueError, selection.blast_filter)
    sequence_frame = pd.DataFrame({'Unnamed: 0': ['0'], 'sequence': ['ACGT']})
    for ids in (['unknown'], ['0', '0']):
        with patch.object(selection.pd, 'read_csv', side_effect=[pd.DataFrame({'Query ID': ids}), sequence_frame]):
            rejected(ValueError, selection.select_blast)
    valid = pd.DataFrame([0.1, 0.2])
    for invalid in (pd.DataFrame([0.1]), pd.DataFrame([[1, 2], [3, 4]]),
                    pd.DataFrame([np.nan, 1]), pd.DataFrame([np.inf, 1]), pd.DataFrame(['invalid', '1'])):
        with patch.object(selection.pd, 'read_csv', side_effect=[pd.DataFrame(['ACGT', 'TGCA']), invalid, valid, valid]):
            rejected(ValueError, selection.sort_promoters)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work-dir', type=Path, required=True)
    args = parser.parse_args()
    args.work_dir.mkdir(parents=True, exist_ok=False)
    check_classification(args.work_dir)
    check_selection(args.work_dir)
    result = dict(status='passed', checks=['configured_label_types', 'single_class_auroc',
                  'strict_checkpoint_aliases', 'test_and_predict_paths', 'prediction_alignment',
                  'reporter_boundary', 'selection_row_identity', 'successful_no_hit',
                  'failed_blast_stops', 'fresh_output_and_space_paths'],
                  real_checkpoint_inference=False, real_blast_search=False)
    (args.work_dir / 'checks.json').write_text(json.dumps(result, indent=2) + '\n')
    print('All legacy CPU checks passed; no checkpoint, GPU, or BLAST database was required.')


if __name__ == '__main__':
    main()
