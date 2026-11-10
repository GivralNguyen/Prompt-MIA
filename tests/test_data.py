"""Check dataset selection and client partitions without downloading data."""

import csv
import json
import numpy as np
from PIL import Image
import unittest
from unittest.mock import patch
from pathlib import Path
from tempfile import TemporaryDirectory
import torch
import torchvision.transforms as T

from promptmia.data import (ImageCache, Images, NORM_STATS, balanced_indices,
                            client_split, dirichlet_split, extreme_split)


def _assert_partition(parts, size):
    combined = np.concatenate(parts)
    assert np.array_equal(np.sort(combined), np.arange(size))
    assert len(set(combined)) == size


class DataTests(unittest.TestCase):
    def test_class_quotas_never_duplicate_rows_from_small_classes(self):
        labels = np.repeat(np.arange(7), [4, 10, 10, 10, 10, 10, 10])
        indices = balanced_indices(labels, 7, 56, seed=3)
        assert len(indices) == 52
        assert len(np.unique(indices)) == 52
        assert np.array_equal(np.bincount(labels[indices]), [4, 8, 8, 8, 8, 8, 8])
        assert len(np.unique(indices[labels[indices] == 0])) == 4
        assert np.array_equal(indices, balanced_indices(labels, 7, 56, seed=3))


    def test_evaluation_subset_does_not_repeat_samples(self):
        labels = np.repeat(np.arange(7), 400)
        rows = balanced_indices(labels, 7, 2506, seed=0)
        assert len(rows) == len(set(rows)) == 2506
        counts = np.bincount(labels[rows])
        assert np.array_equal(counts, np.full(7, 358))


    def test_dirichlet_partition_is_disjoint_complete_and_deterministic(self):
        labels = np.repeat(np.arange(10), 100)
        parts = dirichlet_split(labels, 5, 0.5, seed=7, min_size=20)
        _assert_partition(parts, len(labels))
        repeated = dirichlet_split(labels, 5, 0.5, seed=7, min_size=20)
        assert all(np.array_equal(a, b) for a, b in zip(parts, repeated))
        with self.assertRaisesRegex(ValueError, "at least"):
            dirichlet_split(labels, 80, 0.5, seed=7)
        with self.assertRaisesRegex(ValueError, "positive"):
            dirichlet_split(labels, 5, float("nan"), seed=7)


    def test_extreme_partition_preserves_rows_and_dominant_classes(self):
        labels = np.repeat(np.arange(10), 1000)
        parts = extreme_split(labels, 20, 1, seed=4)
        _assert_partition(parts, len(labels))
        for part in parts:
            counts = np.bincount(labels[part], minlength=10)
            assert counts.max() / len(part) > 0.98
        repeated = extreme_split(labels, 20, 1, seed=4)
        assert all(np.array_equal(a, b) for a, b in zip(parts, repeated))


    def test_four_sources_stay_in_separate_clients(self):
        source = np.repeat(np.arange(4), 400)
        labels = np.tile(np.repeat(np.arange(4), 100), 4)
        parts = client_split(dict(train_labels=labels, train_source=source), 8, 0.5, seed=2)
        _assert_partition(parts, len(labels))
        assert all(len(np.unique(source[part])) == 1 for part in parts)


    def test_cpu_image_cache_matches_standard_normalization(self):
        array = np.arange(8 * 8 * 3, dtype=np.uint8).reshape(8, 8, 3)
        dataset = Images(lambda _: Image.fromarray(array), [2], transform=T.PILToTensor())
        cache = ImageCache(dataset, "cpu", cache_device="cpu")
        actual, labels = cache.get(torch.tensor([0]))
        expected = T.Normalize(*NORM_STATS)(T.ToTensor()(Image.fromarray(array)))
        assert torch.equal(actual[0], expected)
        assert labels.tolist() == [2]


    def test_food101_manifest_pairs_images_text_and_labels(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        tmp_path = Path(temporary.name)
        from promptmia.vilt import Food101Dataset
        directory = tmp_path / "food101"
        directory.mkdir()
        for i in range(2):
            Image.new("RGB", (400, 400), color=(i * 100, 0, 0)).save(directory / f"{i}.png")
        with (directory / "train.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=["image", "text", "label"])
            writer.writeheader()
            for i in range(2):
                writer.writerow(dict(image=f"{i}.png", text=f"dish {i}", label=i))
        dataset = Food101Dataset(tmp_path, "train", num_classes=2)
        image, text, label = dataset.sample(1)
        assert image.shape == (3, 384, 384)
        assert text == "dish 1" and label == 1
        text_image, _, _ = dataset.sample(0, "text")
        assert torch.equal(text_image, torch.ones_like(text_image))

    def test_vilt_round_saves_complete_records_and_resolves_project_paths(self):
        from promptmia.vilt import EmbeddingCache, train_vilt

        class SmallBackbone(torch.nn.Module):
            def __init__(self, checkpoint):
                super().__init__()
                self.pooler = torch.nn.Linear(768, 768)
                self.requires_grad_(False)

            def blocks(self, sequence, mask):
                return sequence + sequence.mean(dim=1, keepdim=True)

        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        generator = torch.Generator().manual_seed(1)

        def cache(size):
            return EmbeddingCache(torch.randn(size, 3, 768, generator=generator),
                                  torch.ones(size, 3, dtype=torch.long), torch.arange(size) % 2)

        config = dict(name="vilt/check", rounds=1, clients=2, clients_per_round=2,
                      local_epochs=1, batch_size=16, M=5, N=2, num_classes=2,
                      model_root="weights", data_root="inputs", defense="none")
        previous_threads = torch.get_num_threads()
        try:
            torch.set_num_threads(2)
            with patch("promptmia.vilt.ViLTBackbone", side_effect=SmallBackbone) as constructor, \
                    patch("promptmia.vilt.load_food101", return_value=(cache(96), cache(8))) as loader:
                run = train_vilt(config, root, "cpu")
            self.assertEqual(constructor.call_args.args[0], root / "weights/vilt_200k_mlm_itm.ckpt")
            self.assertEqual(loader.call_args.kwargs["root"], root / "inputs")
            self.assertEqual(loader.call_args.kwargs["tokenizer_path"], root / "weights/bert-base-uncased")
            metrics = json.loads((run / "metrics.json").read_text())
            self.assertTrue(metrics["completed"])
            self.assertIn("1", metrics["test_acc"])
            self.assertTrue((run / "state.pt").exists())
            self.assertTrue((run / "ckpt_round1.pt").exists())
            queries = torch.load(root / "cache/queries/food101_multimodal_vilt.pt", weights_only=True)
            self.assertEqual(queries["train"].shape, (96, 768))
        finally:
            torch.set_num_threads(previous_threads)

    def test_clip_cache_and_one_round_with_a_small_local_text_tower(self):
        from transformers import CLIPTextConfig, CLIPTextModelWithProjection
        from promptmia.clip import PackedCache, CLIPL2P, train_clip
        with TemporaryDirectory() as directory:
            root = Path(directory)
            folder = root / 'data/fashionmm_clip'
            folder.mkdir(parents=True)
            rng = np.random.default_rng(9)
            for split, count in [('train', 96), ('test', 8)]:
                ids = np.tile(np.array([30, 2, 3, 31]), (count, 1))
                np.savez_compressed(folder / f'{split}.npz', ids=ids,
                    img=rng.normal(size=(count, 512)).astype(np.float32),
                    txt=rng.normal(size=(count, 512)).astype(np.float32),
                    y=np.arange(count) % 2, classes=np.array(['a', 'b']))
            cache = PackedCache('train', 'cpu', root / 'data')
            x, labels = cache.get([0, 1])
            self.assertEqual(x[0].shape, (2, 4))
            self.assertEqual(labels.tolist(), [0, 1])

            def small_model(classes, M, N, model_path=None):
                text_config = CLIPTextConfig(vocab_size=32, hidden_size=512, intermediate_size=64,
                    num_hidden_layers=1, num_attention_heads=8, projection_dim=512,
                    max_position_embeddings=32, bos_token_id=30, eos_token_id=31, pad_token_id=31)
                return CLIPL2P(classes, M, N, text_model=CLIPTextModelWithProjection(text_config))

            model = small_model(2, 5, 2)
            query = model.query(x)
            expected = x[1] / (x[1].norm(dim=1, keepdim=True) + 1e-6)
            expected += x[2] / (x[2].norm(dim=1, keepdim=True) + 1e-6)
            torch.testing.assert_close(query, expected)
            logits, key_loss, _ = model(x, query)
            (torch.nn.functional.cross_entropy(logits, labels) + .5 * key_loss).backward()
            self.assertGreater(model.pool.prompt.grad.abs().sum().item(), 0)
            self.assertTrue(all(p.grad is None for p in model.clip_text.parameters()))
            config = dict(name='clip/check', rounds=1, clients=2, clients_per_round=2,
                          local_epochs=1, batch_size=16, M=5, N=2, save_rounds=[0, 1])
            with patch('promptmia.clip.CLIPL2P', side_effect=small_model):
                run = train_clip(config, root, 'cpu')
            metrics = json.loads((run / 'metrics.json').read_text())
            self.assertTrue(metrics['completed'])
            self.assertTrue((run / 'state.pt').exists())
            saved = torch.load(root / 'cache/queries/fashion_clip.pt', weights_only=True)
            self.assertEqual(saved['train'].shape, (96, 512))
            self.assertEqual(saved['test'].shape, (8, 512))


if __name__ == "__main__":
    unittest.main()
