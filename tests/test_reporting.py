"""Small synthetic checks of measurements and all paper output paths."""

import contextlib
import io
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from promptmia.evaluation import evaluate_job
from promptmia.plotting import plot_artifact, summarize_attack, _table_rows
from promptmia.diagnostics import mad_flags, drift_statistics, distinctiveness_statistics, quintiles
from promptmia.attack import inject_batch
from promptmia.workflow import code_fingerprint


PROJECT = Path(__file__).resolve().parents[1]
DETECTORS = ('IsolationForest', 'LocalOutlierFactor', 'OneClassSVM', 'EllipticEnvelope')


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def attack_result(name, values, batches=(1, 4), dataset='cifar10'):
    return dict(name=name, dataset=dataset, batch_sizes=list(batches), methods={
        'promptmia': {str(b): dict(asr=list(values), advantage=[2*x-1 for x in values]) for b in batches}})


def query_fixture(root):
    run = root / 'runs' / 'fixture'
    config = dict(dataset='fixture', backbone='fixture', N=2, M=8, rounds=60,
                  noise_sigma=.15, query_cache='cache/queries.pt')
    write_json(run / 'config.json', config)
    gen = torch.Generator().manual_seed(17)
    keys = torch.randn(8, 12, generator=gen)
    queries = dict(train=torch.randn(30, 12, generator=gen), test=torch.randn(12, 12, generator=gen))
    torch.save({60: keys}, run / 'keys.pt')
    (root / 'cache').mkdir()
    torch.save(queries, root / 'cache/queries.pt')
    return run, config, queries


class ReportingTests(unittest.TestCase):
    def test_run_means_have_equal_weight(self):
        a = attack_result('a', [.5, .7])
        b = attack_result('b', [1., 1., 1., 1.])
        result = summarize_attack([a, b], 'promptmia', 'asr')
        np.testing.assert_allclose(result['mean'], [.8, .8])
        np.testing.assert_allclose(result['std'], [.2, .2])
        np.testing.assert_allclose(summarize_attack([a], 'promptmia', 'asr')['std'], [.1, .1])
        with self.assertRaises(ValueError):
            summarize_attack([a, attack_result('c', [.8], batches=(1, 8))], 'promptmia', 'asr')

    def test_noise_table_uses_final_round_and_sample_sd(self):
        records = []
        for epsilon in (None, 3, 5, 8):
            for accuracy in (.7, .8, .9):
                records.append(dict(dataset='cifar10', config=dict(dp_eps=epsilon),
                                    metrics=dict(test_acc={'1': .1, '9': .2, '60': accuracy})))
        row = _table_rows(dict(id='table2', ddof=1), records)[0]
        for name in ('eps3', 'eps5', 'eps8', 'non_dp'):
            self.assertAlmostEqual(row[name], 80.)
            self.assertAlmostEqual(row[name+'_std'], 10.)
            self.assertEqual(row[name+'_runs'], 3)

    def test_plain_evaluation_is_repeatable_and_preserves_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = root / 'runs' / 'fixture'
            config = dict(dataset='fixture', backbone='fixture', N=2, rounds=60,
                          query_cache='cache/queries.pt')
            write_json(run / 'config.json', config)
            gen = torch.Generator().manual_seed(8)
            keys = torch.randn(6, 24, generator=gen)
            torch.save({0: keys+1, 60: keys}, run / 'keys.pt')
            queries = dict(train=torch.randn(30, 24, generator=gen), test=torch.randn(12, 24, generator=gen))
            (root / 'cache').mkdir()
            torch.save(queries, root / 'cache' / 'queries.pt')
            checkpoint = (run / 'keys.pt').read_bytes()
            job = dict(name='check/attack', kind='attack', run='fixture', n_targets=8,
                       batch_sizes=[1, 4], repeats=2, methods=['promptmia', 'naive'], chunk=4)
            with contextlib.redirect_stdout(io.StringIO()):
                result = evaluate_job(job, root)
                repeated = evaluate_job(job, root)
            self.assertEqual(result, repeated)
            self.assertEqual(checkpoint, (run / 'keys.pt').read_bytes())
            self.assertEqual(json.loads((root / 'results/check/attack.json').read_text()), result)
            for method in result['methods'].values():
                self.assertEqual(method['1']['tpr'], [1., 1.])
                for game in method.values():
                    for tpr, fpr, advantage, asr in zip(game['tpr'], game['fpr'], game['advantage'], game['asr']):
                        self.assertTrue(0 <= tpr <= 1 and 0 <= fpr <= 1)
                        self.assertAlmostEqual(advantage, tpr-fpr)
                        self.assertAlmostEqual(asr, (1+advantage)/2)

    def test_missing_measurements_raise(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileNotFoundError):
                plot_artifact(dict(id='figure4', sources=['missing']), directory)

    def test_matched_loss_uses_recorded_client_population(self):
        from promptmia.attack import security_game
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run, _, queries = query_fixture(root)
            member_ids = torch.tensor([2, 7, 11, 19])
            loss_pairs = {
                0: (torch.tensor([.1, .2, .3, .4]), torch.linspace(.2, .9, 12)),
                1: (torch.tensor([.2, .4, .6, .8]), torch.linspace(.1, 1., 12)),
                2: (torch.tensor([.5, .4, .3, .2]), torch.linspace(.3, .8, 12))}
            torch.save(dict(target=0, member_index=member_ids, local={1: loss_pairs},
                            glob={1: loss_pairs[0]}), run / 'mia_losses.pt')
            job = dict(name='client_attack', kind='attack', run='fixture', client_population=True,
                       batch_sizes=[4], methods=['promptmia'], repeats=2)
            with patch('promptmia.attack.security_game', wraps=security_game) as game, contextlib.redirect_stdout(io.StringIO()):
                measured = evaluate_job(job, root)
            self.assertEqual(measured['member_targets'], len(member_ids))
            for call in game.call_args_list:
                self.assertTrue(torch.equal(call.args[1], queries['train'][member_ids]))
            loss = evaluate_job(dict(name='loss', kind='loss', run='fixture',
                                     attack_source='client_attack', batch_size=4), root)
            self.assertAlmostEqual(loss['matched_fpr'], np.mean(measured['methods']['promptmia']['4']['fpr']))
            for method in ('blackbox_loss', 'loss_series', 'fedmia_i'):
                self.assertLessEqual(loss['methods'][method]['at_fpr']['fpr'], loss['matched_fpr'])

    def test_dropout_uses_half_of_injected_keys(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            query_fixture(root)
            job = dict(name='dropout', kind='attack', run='fixture', prompt_dropout=.4,
                       n_targets=12, batch_sizes=[1, 4], methods=['promptmia'], repeats=2)
            with contextlib.redirect_stdout(io.StringIO()):
                half = evaluate_job(job, root)
                all_keys = evaluate_job(dict(job, name='dropout_all', min_selected=2), root)
            self.assertEqual(half['min_selected'], 1)
            self.assertEqual(all_keys['min_selected'], 2)
            for batch in ('1', '4'):
                for metric in ('tpr', 'fpr'):
                    a = np.asarray(half['methods']['promptmia'][batch][metric])
                    b = np.asarray(all_keys['methods']['promptmia'][batch][metric])
                    self.assertTrue(np.all(a >= b))
            self.assertGreater(np.mean(half['methods']['promptmia']['1']['tpr']),
                               np.mean(all_keys['methods']['promptmia']['1']['tpr']))

    def test_adaptive_noise_applies_eot_to_training_and_test_targets(self):
        from promptmia.attack import robust_inject
        class SmallQueryModel:
            def query(self, images):
                return images.flatten(1)
        class SmallImageCache:
            def __init__(self, dataset, device):
                self.images, self.labels = dataset
                self.device = torch.device(device)
            def __len__(self):
                return len(self.images)
            def get(self, indices):
                return self.images[indices], self.labels[indices]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, _, queries = query_fixture(root)
            data = {split: (q.reshape(-1, 3, 2, 2), torch.zeros(len(q), dtype=torch.long))
                    for split, q in queries.items()}
            job = dict(name='adaptive', kind='attack', run='fixture', defense='noise', eot=8,
                       eot_coverage=.95, images_per_chunk=16, n_targets=6,
                       batch_sizes=[1, 4], methods=['promptmia'], repeats=1)
            with patch('promptmia.evaluation._load_vision_model', return_value=SmallQueryModel()), \
                    patch('promptmia.data.load_dataset', return_value=data), \
                    patch('promptmia.data.ImageCache', SmallImageCache), \
                    patch('promptmia.attack.robust_inject', wraps=robust_inject) as inject, \
                    contextlib.redirect_stdout(io.StringIO()):
                measured = evaluate_job(job, root)
                repeated = evaluate_job(job, root)
            self.assertEqual(measured, repeated)
            self.assertGreaterEqual(inject.call_count, 4)
            for call in inject.call_args_list:
                self.assertEqual(call.args[2].shape[1], 8)
                self.assertEqual(call.kwargs['coverage'], .95)
            self.assertEqual(measured['member_targets'], 6)
            self.assertEqual(measured['nonmember_targets'], 6)

    def test_drift_audit_counts_and_threshold(self):
        np.testing.assert_array_equal(mad_flags([0, 0, 1, 2, 2], 1), np.zeros(5, dtype=bool))
        np.testing.assert_array_equal(mad_flags([1, 2, 3, 4, 20], 2), [False, False, False, False, True])
        gen = torch.Generator().manual_seed(9)
        keys = torch.randn(8, 12, generator=gen)
        series = {r: keys + .01*r*torch.randn(8, 12, generator=gen) for r in range(7)}
        queries = torch.randn(16, 12, generator=gen)
        a = drift_statistics(series, queries, n_targets=4, N=2)
        self.assertEqual(a, drift_statistics(series, queries, n_targets=4, N=2))
        for rate in a['pooled'].values():
            for measured in rate.values():
                tp, fp, fn = (measured[k] for k in ('tp', 'fp', 'fn'))
                self.assertAlmostEqual(measured['f1'], 2*tp/(2*tp+fp+fn))
        with self.assertRaises(ValueError):
            drift_statistics({0: keys, 6: keys}, queries, n_targets=4, N=2)

    def test_drift_injected_keys_use_previous_round_reference(self):
        gen = torch.Generator().manual_seed(1)
        M, D, n_targets, N, dmin, delta, seed = 16, 8, 8, 3, .02, .05, 0
        base = torch.randn(M, D, generator=gen)
        series = {r: base + .02 * r * torch.randn(M, D, generator=gen) for r in range(6)}
        series[6] = series[5] + .15 * torch.randn(M, D, generator=gen)     # final round != previous round
        queries = base[torch.randint(M, (24,), generator=gen)] + .05 * torch.randn(24, D, generator=gen)  # near real keys
        result = drift_statistics(series, queries, n_targets=n_targets, N=N, delta_min=dmin, delta=delta,
                                  seed=seed, steady_round=5)
        final, previous = series[6], series[5].double()
        ids = np.random.default_rng(seed).choice(len(queries), n_targets, replace=False)
        pools, slots = inject_batch(final, queries[ids].float(), N, dmin, delta,
                                    torch.Generator().manual_seed(seed), evade='norm')
        drift_prev = (pools.double() - previous.unsqueeze(0)).norm(dim=2).numpy()
        drift_final = (pools.double() - final.double().unsqueeze(0)).norm(dim=2).numpy()

        def counts(ref):
            out = {}
            for c in (1, 2, 3):
                tp = fp = fn = 0
                for t in range(n_targets):
                    mask = np.zeros(M, bool); mask[slots[t].numpy()] = True
                    flagged = mad_flags(ref[t], c)
                    tp += int(flagged[mask].sum()); fp += int(flagged[~mask].sum()); fn += int((~flagged[mask]).sum())
                out[c] = (tp, fp, fn)
            return out

        got = {c: (result['injection_counts'][str(c)]['tp'], result['injection_counts'][str(c)]['fp'],
                   result['injection_counts'][str(c)]['fn']) for c in (1, 2, 3)}
        prev_counts, final_counts = counts(drift_prev), counts(drift_final)
        self.assertEqual(got, prev_counts)
        self.assertNotEqual(prev_counts, final_counts)

    def test_embedding_query_source(self):
        from promptmia.evaluation import _embedding
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            query_fixture(root)
            job = dict(name='x', kind='embedding', run='fixture', n_queries=20)
            with self.assertRaisesRegex(ValueError, 'CUDA'):
                _embedding(job, root, 'cpu')
            with self.assertRaisesRegex(ValueError, 'first training images'):
                _embedding(dict(job, query_sampling='random'), root, 'cpu')
            with self.assertRaisesRegex(ValueError, 'unknown query_source'):
                _embedding(dict(job, query_source='other'), root, 'cpu')
            cached = _embedding(dict(job, query_source='cache'), root, 'cpu')
            self.assertEqual((cached['n_keys'], cached['n_queries']), (8, 20))
            keys_only = _embedding(dict(job, include_queries=False), root, 'cpu')
            self.assertEqual(keys_only['n_queries'], 0)

    def test_figures_11_and_15_use_cached_queries(self):
        using = {job['name'] for job in json.loads((PROJECT / 'configs' / 'paper.json').read_text())['evaluation']
                 if job.get('query_source') == 'cache'}
        self.assertEqual(using, {f'extreme_embedding/cifar100_M{m}' for m in (10, 20, 30, 40)} | {'outlier_embedding/promptmia'})

    def test_diagnostics_is_in_evaluation_fingerprint(self):
        # Editing the drift / distinctiveness detector must invalidate cached Table 4 / Table 7 evaluations.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shutil.copytree(PROJECT / 'promptmia', root / 'promptmia')
            before = code_fingerprint(root, 'evaluation')
            target = root / 'promptmia' / 'diagnostics.py'
            target.write_text(target.read_text() + '\n# detector change\n')
            self.assertNotEqual(before, code_fingerprint(root, 'evaluation'))

    def test_distinctiveness_bins_and_client_batches(self):
        bins = quintiles(np.arange(10)[::-1], np.ones(10), np.arange(10)[::-1]/10)
        np.testing.assert_allclose([b['fpr'] for b in bins], [.05, .25, .45, .65, .85])
        gen = torch.Generator().manual_seed(13)
        keys = torch.randn(8, 12, generator=gen)
        train = torch.randn(24, 12, generator=gen)
        test = torch.randn(10, 12, generator=gen)
        clients = [np.arange(12), np.arange(12, 24), np.arange(2)]
        result = distinctiveness_statistics(keys, train, test, clients, n_targets=10, batch_sizes=[3, 5],
                                            n_batches=3, N=2, chunk=4)
        self.assertEqual(result['excluded_clients'], {'3': 1, '5': 1})
        for batch, measured in result['games'].items():
            np.testing.assert_array_equal(measured['tpr'], np.ones(10))
            self.assertEqual(sum(b['targets'] for b in result['quintiles'][batch]), 10)

    def test_figures_2_and_3_write_one_image_per_subfigure(self):
        plan = json.loads((PROJECT / 'configs' / 'paper.json').read_text())
        rng = np.random.default_rng(1)
        with tempfile.TemporaryDirectory() as directory, threadpool_limits(limits=1):
            root = Path(directory)
            for job in plan['evaluation']:
                if job['name'].startswith('keys/') or job['name'] == 'embedding/cifar10':
                    count = 20 if job.get('include_queries') is False else 70
                    injected = [] if job.get('variant', 'benign') == 'benign' else [0, 1, 2, 3]
                    write_json(root / 'results' / (job['name']+'.json'),
                               dict(name=job['name'], coordinates=rng.normal(size=(count, 2)).tolist(), n_keys=20,
                                    injected_slots=injected))
            for spec in plan['artifacts']:
                if spec['id'] in ('figure2', 'figure3'):
                    paths = plot_artifact(spec, root)
                    names = {path.name for path in paths}
                    self.assertEqual(names, {spec['id']+'.json'} | {f"{spec['id']}{letter}.{kind}" for letter in 'abc'
                                                                  for kind in ('png', 'pdf')})
                    self.assertTrue(all(path.is_file() for path in paths))

    def test_every_manifest_artifact_renders(self):
        """Exercise all 30 output schemas without training or downloading data."""
        plan = json.loads((PROJECT / 'configs' / 'paper.json').read_text())
        configs = {config['name']: config for config in plan['training']}
        rng = np.random.default_rng(42)
        with tempfile.TemporaryDirectory() as directory, threadpool_limits(limits=1):
            root = Path(directory)
            shutil.copytree(PROJECT / 'assets', root / 'assets')
            for config in configs.values():
                run = root / 'runs' / config['name']
                write_json(run / 'config.json', config)
                write_json(run / 'metrics.json', dict(test_acc={'1': .2, '60': .7+.01*config.get('seed', 0)}))
            for job in plan['evaluation']:
                config = configs.get(job.get('run'), {})
                record = dict(name=job['name'], kind=job['kind'], run=job.get('run'), settings=job, config=config,
                              dataset=config.get('dataset'), backbone=config.get('backbone'), round=job.get('round', 60))
                if job['kind'] == 'attack':
                    batches = job['batch_sizes']
                    record.update(batch_sizes=batches, methods={m: {
                        str(b): dict(tpr=[.9, .95], fpr=[.1, .15], advantage=[.8, .8], asr=[.9, .9])
                        for b in batches} for m in job.get('methods', ['promptmia'])})
                elif job['kind'] == 'loss':
                    record['methods'] = {m: dict(asr=.7, advantage=.4, at_fpr=dict(asr=.65, advantage=.3)) for m in
                                         ('blackbox_loss', 'loss_series', 'fedmia_i')}
                    record['methods']['promptmia'] = dict(asr=.9, advantage=.8, tpr=.95, fpr=.15)
                elif job['kind'] == 'outlier':
                    record['methods'] = {m: dict(precision=.4, recall=.2, f1=.8/3) for m in DETECTORS}
                elif job['kind'] == 'bound':
                    record['log10_mean_bound'] = -18.
                elif job['kind'] == 'embedding':
                    nk = config.get('M', 20)
                    nq = 50 if job.get('include_queries', True) else 0
                    record.update(coordinates=rng.normal(size=(nk+nq, 2)).tolist(), n_keys=nk, n_queries=nq,
                                  injected_slots=[] if job.get('variant', 'benign') == 'benign' else [0, 1, 2, 3],
                                  flagged={m: [1, 3] for m in DETECTORS})
                elif job['kind'] == 'sparsity':
                    record['rows'] = [dict(model=m, total_params=1000000, modified_params=3072, sparsity_percent=.3072)
                                      for m in job['backbones']]
                elif job['kind'] == 'drift':
                    record['pooled'] = {rate: {str(c): dict(f1=.2) for c in (1, 2, 3)} for rate in ('1/10', '1/60')}
                elif job['kind'] == 'distinctiveness':
                    record['quintiles'] = {str(b): [dict(fpr=.1, tpr=1., asr=.95) for _ in range(5)] for b in (64, 256)}
                else:
                    self.fail('No fixture for '+job['kind'])
                write_json(root / 'results' / (job['name']+'.json'), record)
            expected = {f'figure{i}' for i in range(1, 22)} | {f'table{i}' for i in range(1, 10)}
            self.assertEqual({a['id'] for a in plan['artifacts']}, expected)
            for spec in plan['artifacts']:
                with self.subTest(artifact=spec['id']):
                    paths = plot_artifact(spec, root)
                    panels = {'figure2': 3, 'figure3': 3, 'figure10': 2, 'figure15': 4, 'figure16': 4,
                              'figure17': 6, 'figure18': 4}.get(spec['id'], 1)
                    self.assertEqual(len(paths), 2*panels+1)
                    for path in paths:
                        self.assertTrue(path.is_file())
                        self.assertGreater(path.stat().st_size, 20)
                    summary = json.loads(next(p for p in paths if p.suffix == '.json').read_text())
                    self.assertEqual(summary['id'], spec['id'])
                    self.assertEqual(summary['sources'], spec['sources'])
                    if spec.get('panels'):
                        self.assertEqual(len(summary['panels']), len(spec['panels']))


if __name__ == '__main__':
    unittest.main()
