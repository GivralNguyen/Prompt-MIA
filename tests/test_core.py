"""Small CPU checks for geometry, gradient routing, and client isolation."""
import unittest
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from promptmia.attack import gen_key_with_similarity, inject_batch, all_injected_selected, security_game, robust_inject
from promptmia.fl import fedavg, local_train, run_fedavg
from promptmia.model import PromptPool, L2PViT


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = PromptPool(M=5, N=2, dim=3)
        self.head = nn.Linear(3, 2)

    trainable_state = L2PViT.trainable_state
    load_trainable_state = L2PViT.load_trainable_state

    def forward(self, x, q=None):
        selected, loss = self.pool.select(x if q is None else q)
        return self.head(x + self.pool.prompt[selected].mean(1)), loss, selected


class TinyCache:
    def __init__(self):
        self.device = torch.device('cpu')
        self.images = torch.tensor([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.],
                                    [1., 1., 0.], [0., 1., 1.], [1., 0., 1.]])
        self.labels = torch.tensor([0, 1, 0, 1, 1, 0])

    def __len__(self):
        return len(self.images)

    def get(self, ids):
        return self.images[ids], self.labels[ids]


class TinyDataset:
    def __init__(self):
        self.labels = TinyCache().labels.numpy()
        self.source_indices = np.arange(len(self.labels))

    def __len__(self):
        return len(self.labels)


class CoreTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)

    def test_key_geometry_and_untouched_slots(self):
        query = torch.randn(32, dtype=torch.float64) * 3
        key = gen_key_with_similarity(query, .63, torch.Generator().manual_seed(4))
        self.assertAlmostEqual(F.cosine_similarity(query, key, dim=0).item(), .63, places=12)
        self.assertAlmostEqual(query.norm().item(), key.norm().item(), places=12)
        keys, targets = torch.randn(20, 32), torch.randn(11, 32)
        pools, slots = inject_batch(keys, targets, 4, .02, .05, torch.Generator().manual_seed(3))
        self.assertTrue(all_injected_selected(pools, slots, targets[:, None]).all())
        mask = torch.ones(11, 20, dtype=torch.bool).scatter_(1, slots, False)
        self.assertTrue(torch.equal(pools[mask], keys.expand(11, -1, -1)[mask]))

    def test_invalid_geometry_and_batch(self):
        with self.assertRaises(ValueError):
            gen_key_with_similarity(torch.ones(3), 1.1, torch.Generator())
        with self.assertRaises(ValueError):
            security_game(torch.randn(5, 3), torch.randn(4, 3), torch.randn(2, 3), 2, 5, .02, .05, seed=0)

    def test_only_selected_keys_receive_gradients(self):
        pool = PromptPool(M=8, N=2, dim=5)
        selected, loss = pool.select(torch.randn(1, 5))
        loss.backward()
        mask = torch.ones(8, dtype=torch.bool)
        mask[selected.flatten()] = False
        self.assertTrue(torch.equal(pool.key.grad[mask], torch.zeros_like(pool.key.grad[mask])))
        self.assertGreater(pool.key.grad[selected.flatten()].abs().sum().item(), 0)
        self.assertEqual(set(dict(pool.named_parameters())), {'prompt', 'key'})

    def test_weighted_aggregation_has_independent_storage(self):
        states = [{'x': torch.tensor([1., 3.])}, {'x': torch.tensor([5., 7.])}]
        result = fedavg(states, [1, 3])
        torch.testing.assert_close(result['x'], torch.tensor([4., 6.]))
        result['x'].zero_()
        torch.testing.assert_close(states[0]['x'], torch.tensor([1., 3.]))

    def test_clients_start_from_the_same_global_state(self):
        cache, model = TinyCache(), TinyModel()
        initial = model.trainable_state()
        clients = [np.array([0, 1]), np.array([2, 3, 4, 5])]
        cfg = dict(seed=5, rounds=1, clients_per_round=2, local_epochs=1, batch_size=2,
                   lr=1e-3, key_lr=1e-3, lam=.5)
        gen, expected = torch.Generator().manual_seed(5), []
        for indices in clients:
            model.load_trainable_state(initial)
            local_train(model, cache, cache.images, indices, 1, 2, 1e-3, .5, gen, 1e-3)
            expected.append(model.trainable_state())
        reference = fedavg(expected, [2, 4])
        model.load_trainable_state(initial)
        actual = run_fedavg(model, cache, cache.images, cache, cache.images, clients, cfg)
        for name in reference:
            torch.testing.assert_close(actual[name], reference[name])
        replacement = dict(actual)
        replacement['pool.key'] = actual['pool.key'][:3].clone()
        replacement['pool.prompt'] = actual['pool.prompt'][:3].clone()
        model.load_trainable_state(replacement)
        self.assertEqual(model.pool.key.shape, (3, 3))

    def test_frequency_penalty_changes_selection_and_keeps_cosine_key_loss(self):
        pool = PromptPool(M=3, N=1, dim=2)
        with torch.no_grad():
            pool.key.copy_(torch.tensor([[1., 0.], [.8, .6], [-1., 0.]]))
        pool.freq = torch.tensor([10., 0., 0.])
        selected, key_loss = pool.select(torch.tensor([[1., 0.]]))
        self.assertEqual(selected.item(), 1)
        self.assertAlmostEqual(key_loss.item(), .2, places=6)
        torch.testing.assert_close(pool.freq, torch.tensor([10., 1., 0.]))
        pool.eval()
        selected, _ = pool.select(torch.tensor([[1., 0.]]))
        self.assertEqual(selected.item(), 0)

    def test_norm_rescaling_preserves_selection_geometry(self):
        keys, target = torch.randn(20, 32), torch.randn(7, 32)
        base, slots = inject_batch(keys, target, 4, .02, .05, torch.Generator().manual_seed(5))
        scaled, scaled_slots = inject_batch(keys, target, 4, .02, .05,
                                            torch.Generator().manual_seed(5), evade='norm')
        torch.testing.assert_close(slots, scaled_slots)
        torch.testing.assert_close(F.normalize(base, dim=2), F.normalize(scaled, dim=2))
        actual = scaled.gather(1, slots[..., None].expand(-1, -1, 32)).norm(dim=2)
        torch.testing.assert_close(actual, keys[slots].norm(dim=2))

    def test_adaptive_keys_cover_independently_transformed_target_copies(self):
        keys, targets = torch.randn(20, 32), torch.randn(5, 32)
        copies = targets[:, None] + .2 * torch.randn(5, 32, 32)
        pools, slots, chosen, calibrated, achieved = robust_inject(keys, targets, copies, 4, .02, .05,
                                                                   torch.Generator().manual_seed(3))
        selected = all_injected_selected(pools.repeat_interleave(32, 0), slots.repeat_interleave(32, 0),
                                         copies.flatten(0, 1)[:, None]).reshape(5, 32)
        self.assertTrue((selected.double().mean(1) >= .95).all())
        self.assertTrue(torch.isfinite(chosen).all())
        # When every target meets coverage, calibration is reported as a success and achieved >= requested.
        self.assertTrue(calibrated.all())
        self.assertTrue((achieved >= .95).all())
        norms = pools.gather(1, slots[..., None].expand(-1, -1, 32)).norm(dim=2)
        torch.testing.assert_close(norms, targets.norm(dim=1)[:, None].expand(-1, 4))

    def test_training_writes_measured_artifacts_and_validates_query_cache(self):
        from promptmia import training
        dataset = TinyDataset()
        data = dict(train=dataset, test=dataset, train_labels=dataset.labels,
                    n_classes=2, fingerprint='input-version-a')
        cfg = dict(name='tiny', rounds=1, clients=2, clients_per_round=2,
                   local_epochs=1, batch_size=2, M=5, N=2, mia_target=0,
                   inject_round=1, inject_target=0, save_rounds=[0, 1])
        clients = [np.array([0, 1]), np.array([2, 3, 4, 5])]
        with TemporaryDirectory() as folder, \
             patch('promptmia.data.load_dataset', return_value=data), \
             patch('promptmia.data.ImageCache', side_effect=lambda *args, **kwargs: TinyCache()), \
             patch('promptmia.data.client_split', return_value=clients), \
             patch('promptmia.data.compute_queries', side_effect=lambda model, cache: cache.images.clone()) as queries, \
             patch('promptmia.training.L2PViT', side_effect=lambda *args, **kwargs: TinyModel()):
            run = training.train(cfg, Path(folder), device='cpu')
            self.assertEqual(queries.call_count, 2)
            metrics = json.loads((run / 'metrics.json').read_text())
            self.assertTrue(metrics['completed'])
            self.assertGreaterEqual(metrics['test_acc']['1'], 0)
            self.assertLessEqual(metrics['test_acc']['1'], 1)
            for artifact in ('state.pt', 'clients.pt', 'keys.pt', 'mia_losses.pt',
                             'ckpt_round0.pt', 'ckpt_round1.pt', 'injection.json'):
                self.assertTrue((run / artifact).is_file(), artifact)
            keys = torch.load(run / 'keys.pt', weights_only=False)
            state = torch.load(run / 'state.pt', weights_only=False)
            self.assertEqual(set(keys), {0, 1})
            torch.testing.assert_close(keys[1], state['pool.key'])
            losses = torch.load(run / 'mia_losses.pt', weights_only=False)
            self.assertEqual(set(losses['local'][1]), {0, 1})
            training.train(dict(cfg, name='same-inputs'), Path(folder), device='cpu')
            self.assertEqual(queries.call_count, 2)
            data['fingerprint'] = 'input-version-b'
            training.train(dict(cfg, name='changed-inputs'), Path(folder), device='cpu')
            self.assertEqual(queries.call_count, 4)


if __name__ == '__main__':
    unittest.main()
